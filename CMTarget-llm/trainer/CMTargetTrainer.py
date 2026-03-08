from torch import nn
import torch
import torch.optim as optim
from tqdm import tqdm

from embedding.FeatureExtract import FeatureExtractor
from embedding.dataset_build import *
from model.CMTargetModel import CMTargetModel
from model.multi_fusion import *
from model.moe import *
from utils.metrix import *
from utils.utils import TrainLogger, PredictorLogger



class CMTargetTrainer():
    """
    input:
        dataloader: (compound, protein, label), [3, batch_size, token_num, token_dim]
    
    """
    def __init__(self, configs, encoder_path):
        self.configs = configs
        self.device = configs['device']
        self.model = self.get_model()
        self.encoder_path = encoder_path
        # self.dataloader = dataloader

        self.learning_rate = configs['learning_rate']
        self.epochs = configs['epochs']
        self.batch_size = configs['batch_size']

        self.criterion = nn.BCELoss()  # 使用二分类交叉熵损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.feature_extractor = FeatureExtractor()

        # 数据loader
        self.train_dataloader = self.get_dataloader(encoder_path)
        self.evl_dataloader = self.get_dataloader(encoder_path)


    def get_dataloader(self, encoder_path):
        train_ds = EncodedDTIDataset(encoder_path)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        return train_loader

    
    def get_model(self):
        model = CMTargetModel(self.configs)
        if self.configs['model_path'] != '':
            print('Get model from:', self.configs['model_path'])
            model.load_model(self.configs['model_path'])

        return model
    

    def model_train_anepoch(self, model, epoch_id):
        "输入是train_dataloader"
        model = model.to(self.device)
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for compound_batch, protein_batch, label_batch in tqdm(self.train_dataloader):        

            # 清空梯度
            self.optimizer.zero_grad()

            # 前向传播：三种模态特征对齐融合+MoE编码 in:[3,2,501,100]  [3,2,68,768]
            # outputs是概率值[batch_size,]
            pred_score, contrastive_Loss, load_balancing_loss = model(protein_batch, compound_batch)
            
            # 计算预测损失  [2]  [2,1]
            # label = label_batch.unsqueeze(1)  # 确保标签是一个列向量
            # label = label.squeeze(1) #label:[2,1] → [2]
            pred_score = pred_score.cpu()
            pred_loss = self.criterion(pred_score, label_batch)

            # 总损失 = 对比损失 + 负载均衡损失 + 预测损失
            loss = contrastive_Loss + load_balancing_loss + pred_loss

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # 计算准确率
            predicted = (pred_score > 0.5).float()  # 将输出转换为0或1
            correct += (predicted == label_batch).sum().item()
            total += label_batch.size(0)

        avg_loss = running_loss / len(self.train_dataloader)
        accuracy = correct / total * 100
        print(f"Epoch [{epoch_id+1}/{self.epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%") 
        return avg_loss



    def model_evaluate_anepoch(self, evl_model, epoch_id):
        "输入是test_dataloader"
        evl_model = evl_model.to(self.device)
        evl_model.eval()

        targets, predicts = list(), list()
        threshold = 0.5
        with torch.no_grad():
            y_true = []
            y_score = []
            i = 1
            total = len(self.evl_dataloader)
            loop = tqdm(self.evl_dataloader, total=total, smoothing=0, mininterval=1.0)

            for compound_batch, protein_batch, label_batch in loop:

                # 预测结果：三种模态特征对齐融合+MoE编码 in:[3,2,501,100]  [3,2,68,768]
                pred_score, contrastive_Loss, load_balancing_loss = evl_model(protein_batch, compound_batch)              
                pred_score = pred_score.cpu()
                pred = torch.where(pred_score > threshold, torch.tensor(1.0), torch.tensor(0.0))
                
                # 预测list 和  真值list
                targets.extend(label_batch.tolist())
                predicts.extend(pred.tolist())
                arr_targets = np.array(targets)
                arr_predicts = np.array(predicts)

                # 评价指标
                recall, precision, f1, accuracy, auc = calculate_metrics(arr_targets, arr_predicts)

                loop.set_description(f'Batch [{i}/{total}]')
                loop.set_postfix(recall=round(recall, 4), precision=round(precision, 4), f1=round(f1, 4),
                                 accuracy=round(accuracy, 4), auc=round(auc, 4))
                i += 1
                y_true += label_batch.tolist()
                y_score += pred_score.tolist()

        return recall, precision, f1, accuracy, auc, y_true, y_score



    def train(self):
        
        # model = self.get_model()
        # model = self.model
        # criterion = self.criterion
        # train_dataloader = self.train_dataloader
        # evl_dataloader = self.evl_dataloader
        # optimizer = self.optimizer
        epoch = self.epochs
        
        max_f1 = 0
        
        logger = TrainLogger(f"Training", self.configs['timestamp'])
        
        # protein_list = list(self.train_dataloader.dataset.user_data)
        # drug_list = list(self.train_dataloader.dataset.item_data)
        
        logger.update_protein_drug(protein_list, drug_list)

        for i in range(epoch):
            print(f"\n the train epoch is : {i} \n")
            loss = self.model_train_anepoch(self.model, i)
            recall, precision, f1, accuracy, auc, y_true, y_score = self.model_evaluate_anepoch(self.model, i)
            
            logger.write(f"Epoch [{i + 1}/{epoch}]: loss = {round(loss, 4)}, recall = {round(recall, 4)}, precision = {round(precision, 4)}, f1 = {round(f1, 4)}, accuracy = {round(accuracy, 4)}, auc = {round(auc, 4)}")
            logger.log_loss(loss)
            logger.log_metrix(recall, precision, f1, accuracy, auc)
            
            if f1 > max_f1:
                logger.update_true_score(y_true, y_score)
                max_f1 = f1
                self.model.save_model()

