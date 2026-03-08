import ast
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from utils.metrix import calculate_metrics
from utils.utils import PredictLogger

from embedding.dataset_build import *
from embedding.FeatureExtract import *
from model.CMTargetModel import *
from model.multi_fusion import *
from model.moe import *
from utils.metrix import *


class CMTargetPredictor():
    def __init__(self, configs, drugbank_encoder_path, shuffle_path):
        self.configs = configs
        self.device = configs['device']
        
        self.pred_dataloader, self.pred_ds_smiles = self.get_dataloader(drugbank_encoder_path, shuffle_path)
        self.model = self.get_model()

        
    
    def get_model(self):
        model = CMTargetModel(self.configs)
        if self.configs['model_path'] != '':
            print('Get model from:', self.configs['model_path'])
            model.load_model(self.configs['model_path'])

        return model

    
    def get_dataloader(self, encoder_path, shuffle_path):
        # [smiles, sequence, label]
        ds_smiles = DTIDataset(shuffle_path)

        # create encoder_feature dataset # [tensor, tensor, label]
        train_ds = EncodedDTIDataset(encoder_path)
        train_loader = DataLoader(train_ds, batch_size=self.configs['batch_size'], shuffle=True, collate_fn=collate_fn)
        return train_loader, ds_smiles
    
    # recall, precision, f1, accuracy, auc, y_true, y_score = self.pred_anepoch(self.model, self.pred_dataloader)
    def pred_anepoch(self, pred_model, pred_dataloader):
        pred_model = pred_model.to(self.device)
        pred_model.eval()
        targets, predicts = list(), list()
        threshold = 0.5
        with torch.no_grad():
            y_true = []
            y_score = []
            i = 1
            total = len(pred_dataloader)
            loop = tqdm(pred_dataloader, total = total, smoothing=0, mininterval=1.0)

            for compound_batch, protein_batch, label_batch in loop:
                #预测结果
                pred_score, _, _ = pred_model(protein_batch, compound_batch)
                pred_score = pred_score.cpu()
                pred = torch.where(pred_score > threshold, torch.tensor(1.0), torch.tensor(0.0))

                # 预测list 和 真值 list
                targets.extend(label_batch.tolist())
                predicts.extend(pred.tolist())
                arr_targets = np.array(targets)
                arr_predicts = np.array(predicts)

                # 评价指标
                recall, precision, f1, accuracy, auc = calculate_metrics(arr_targets, arr_predicts)

                loop.set_description(f'Batch [{i}/{total}]')
                # loop.set_postfix(recall=round(recall, 4), precision=round(precision, 4), f1=round(f1, 4),
                                #  accuracy=round(accuracy, 4), auc=round(auc, 4))
                i += 1
                y_true += label_batch.tolist()
                y_score += pred_score.tolist()


        # return y_true, y_score
        return recall, precision, f1, accuracy, auc, y_true, y_score



    def predict(self):
        # self.pred_dataloader, self.pred_ds_smiles = self.get_dataloader()
        # self.model = self.get_model()

        protein_list = self.pred_ds_smiles.data['protein'].tolist()
        drug_list = self.pred_ds_smiles.data['compound'].tolist()

        logger = PredictLogger(f"Predicting", self.configs['timestamp'])
        logger.update_protein_drug(protein_list, drug_list)
        
        # 评估结果
        recall, precision, f1, accuracy, auc, y_true, y_score = self.pred_anepoch(self.model, self.pred_dataloader)
        
        logger.update_protein_drug(protein_list, drug_list)
        logger.update_true_score(y_true, y_score)
        logger.log_metrix(recall, precision, f1, accuracy, auc)
        