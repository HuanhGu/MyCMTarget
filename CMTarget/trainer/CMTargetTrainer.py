import ast
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from utils.metrix import calculate_metrics
from utils.utils import TrainLogger
from models.CMTargetModel import CMTargetModel
from datasets.CMTargetDataset import *

class CMTargetTrainer():
    def __init__(self, configs):
        self.configs = configs

    def get_model(self):
        model = CMTargetModel(self.configs['emb'], self.configs['meta_dim'], self.configs['score'], self.configs['timestamp'])
        if self.configs['model_path'] != '':
            print('Get model from:', self.configs['model_path'])
            model.load_model(self.configs['model_path'])

        return model

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.configs['lr'], weight_decay=self.configs['wd'])
        return optimizer

    def get_criterion(self):
        criterion = nn.BCELoss()
        return criterion

    def get_dataloader(self):
        train_meta_df = pd.read_csv(f"./data/{self.configs['data_dir']}/train_meta.csv")
        test_meta_df = pd.read_csv(f"./data/{self.configs['data_dir']}/test_meta.csv")
        train_meta_df['pos_seq'] = train_meta_df['pos_seq'].apply(ast.literal_eval)
        test_meta_df['pos_seq'] = test_meta_df['pos_seq'].apply(ast.literal_eval)

        meta_extractor = FeatureExtractor('fusion', 'fusion', self.configs)
        train_meta_dataset = AttFusionMetaDataset(train_meta_df, meta_extractor, his_len=self.configs['his_len'])
        test_meta_dataset = AttFusionMetaDataset(test_meta_df, meta_extractor, his_len=self.configs['his_len'])

        train_meta_dataloader = DataLoader(train_meta_dataset, batch_size=self.configs['batch_size'], shuffle=True, drop_last=True, num_workers=self.configs['num_workers'])
        test_meta_dataloader = DataLoader(test_meta_dataset, batch_size=self.configs['batch_size'], shuffle=True, drop_last=True, num_workers=self.configs['num_workers'])

        return train_meta_dataloader, test_meta_dataloader

    def train_meta(self, data_loader, model, criterion, optimizer, i, epoch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()
        total_loss = 0
        cnt = 0
        loop = tqdm(data_loader, total=len(data_loader), smoothing=0, mininterval=1.0)
        for item_feature, user_feature, his_item_feature, y in loop:
            item_feature = [feature.to(device) for feature in item_feature]
            user_feature = [feature.to(device) for feature in user_feature]
            for j in range(len(his_item_feature)):
                his_item_feature[j] = [feature.to(device) for feature in his_item_feature[j]]
            pred = model((item_feature, user_feature, his_item_feature))
            pred = pred.cpu()
            loss = criterion(pred, y.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
            cnt += 1
            loop.set_description(f'Epoch [{i + 1}/{epoch}]')
            loop.set_postfix(loss=round(total_loss / cnt, 4))

        return total_loss / len(data_loader)

    def evaluate_meta(self, model, data_loader):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        targets, predicts = list(), list()
        threshold = 0.5
        with torch.no_grad():
            y_true = []
            y_score = []
            i = 1
            total = len(data_loader)
            loop = tqdm(data_loader, total=total, smoothing=0, mininterval=1.0)
            for item_feature, user_feature, his_item_feature, y in loop:
                item_feature = [feature.to(device) for feature in item_feature]
                user_feature = [feature.to(device) for feature in user_feature]
                for j in range(len(his_item_feature)):
                    his_item_feature[j] = [feature.to(device) for feature in his_item_feature[j]]
                pred_score = model((item_feature, user_feature, his_item_feature))
                
                pred_score = pred_score.cpu()
                pred = torch.where(pred_score > threshold, torch.tensor(1.0), torch.tensor(0.0))
                targets.extend(y.tolist())
                predicts.extend(pred.tolist())

                arr_targets = np.array(targets)
                arr_predicts = np.array(predicts)

                recall, precision, f1, accuracy, auc = calculate_metrics(arr_targets, arr_predicts)

                loop.set_description(f'Batch [{i}/{total}]')
                loop.set_postfix(recall=round(recall, 4), precision=round(precision, 4), f1=round(f1, 4),
                                 accuracy=round(accuracy, 4), auc=round(auc, 4))
                i += 1
                y_true += y.tolist()
                y_score += pred_score.tolist()

        return recall, precision, f1, accuracy, auc, y_true, y_score

    def train(self):
        model = self.get_model()
        criterion = self.get_criterion()
        train_meta_dataloader, test_meta_dataloader = self.get_dataloader()
        optimizer = self.get_optimizer(model)

        max_f1 = 0
        epoch = self.configs["epoch"]
        logger = TrainLogger(f"Training", self.configs['timestamp'])
        drug_list = list(test_meta_dataloader.dataset.item_data)
        protein_list = list(test_meta_dataloader.dataset.user_data)
        logger.update_protein_drug(protein_list, drug_list)
        for i in range(epoch):
            loss = self.train_meta(train_meta_dataloader, model, criterion, optimizer, i, epoch)
            recall, precision, f1, accuracy, auc, y_true, y_score = self.evaluate_meta(model, test_meta_dataloader)
            logger.write(f"Epoch [{i + 1}/{epoch}]: loss = {round(loss, 4)}, recall = {round(recall, 4)}, precision = {round(precision, 4)}, f1 = {round(f1, 4)}, accuracy = {round(accuracy, 4)}, auc = {round(auc, 4)}")
            logger.log_loss(loss)
            logger.log_metrix(recall, precision, f1, accuracy, auc)
            if f1 > max_f1:
                logger.update_true_score(y_true, y_score)
                max_f1 = f1
                model.save_model()