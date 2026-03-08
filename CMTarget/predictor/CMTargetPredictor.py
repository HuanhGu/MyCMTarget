import ast
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from utils.metrix import calculate_metrics
from utils.utils import PredictLogger
from models.CMTargetModel import CMTargetModel
from datasets.CMTargetDataset import *

class CMTargetPredictor():
    def __init__(self, configs):
        self.configs = configs

    def get_model(self):
        model = CMTargetModel(self.configs['emb'], self.configs['meta_dim'], self.configs['score'], self.configs['timestamp'])
        if self.configs['model_path'] != '':
            print('Get model from:', self.configs['model_path'])
            model.load_model(self.configs['model_path'])

        return model

    def get_dataloader(self):
        # read data
        train_meta_df = pd.read_csv(f"./data/{self.configs['data_dir']}/train_meta.csv")
        test_meta_df = pd.read_csv(f"./data/{self.configs['data_dir']}/test_meta.csv")
        train_meta_df['pos_seq'] = train_meta_df['pos_seq'].apply(ast.literal_eval)
        test_meta_df['pos_seq'] = test_meta_df['pos_seq'].apply(ast.literal_eval)

        # create dataloader
        meta_extractor = FeatureExtractor('fusion', 'fusion', self.configs)

        test_meta_dataset = AttFusionMetaDataset(test_meta_df, meta_extractor)

        test_meta_dataloader = DataLoader(test_meta_dataset, batch_size=self.configs['batch_size'], shuffle=False, drop_last=False, num_workers=self.configs['num_workers'])

        return test_meta_dataloader

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
                
                # 得到预测得分
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

    def predict(self):
        model = self.get_model()
        test_meta_dataloader = self.get_dataloader()
        drug_list = list(test_meta_dataloader.dataset.item_data)
        protein_list = list(test_meta_dataloader.dataset.user_data)
        logger = PredictLogger(f"Predicting", self.configs['timestamp'])
        logger.update_protein_drug(protein_list, drug_list)
        
        # 评估结果
        recall, precision, f1, accuracy, auc, y_true, y_score = self.evaluate_meta(model, test_meta_dataloader)
        
        logger.update_protein_drug(protein_list, drug_list)
        logger.update_true_score(y_true, y_score)
        logger.log_metrix(recall, precision, f1, accuracy, auc)