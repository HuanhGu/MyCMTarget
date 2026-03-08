import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from embedding.FeatureExtract import *
from torch import nn
from tqdm import tqdm
from pathlib import Path


# --- 重点：自定义 collate_fn 来处理不同长度的 batch ---
def collate_fn(batch):
    drugs, proteins, labels = zip(*batch)
    # 将一个 batch 内的序列 pad 到当前 batch 的最大长度
    drugs_pad = pad_sequence(drugs, batch_first=True)
    proteins_pad = pad_sequence(proteins, batch_first=True)
    labels_tensor = torch.stack(labels)
    return drugs_pad, proteins_pad, labels_tensor




class DTIDataset(Dataset):
    '''
    输入： 
        file_path的三元组内容 : (compound, protein, label)
    输出：
        dataloader形式 
    '''
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        compound = row["compound"]
        protein = row["protein"]
        label = torch.tensor(row["label"], dtype=torch.float32)

        return compound, protein, label



# 为什么我要把这一部分单独摘出来？
#     因为有了序列后, encoder部分可以提前处理, 这样节约训练的时间
class SequencePreEncoder:
    '''
    对蛋白质和化合物序列进行编码

    输入:
        dataloader : (protein_sequence, compound_smiles, label)
        savepath : the savepath of encoder_feature

    输出:
        savepath是encoder的特征向量
    '''
    def __init__(self, dataloader, savepath):
        self.dataloader = dataloader
        self.save_path = savepath
        self.feature_extractor = FeatureExtractor()

    def encode_and_save(self):
        "将数据集所有蛋白质和化合物序列进行编码, 并保存"
        all_protein_features = []
        all_drug_features = []
        all_labels = []


        print(f"🚀 开始预编码特征 (设备: {self.feature_extractor.device})...")
        for compound_batch, protein_batch, label_batch in tqdm(self.dataloader):
            # 1. 提取蛋白质 (返回的是 list of Tensors)
            p_feats = self.feature_extractor.pro_fea_extract(protein_batch)
            all_protein_features.extend(p_feats)

            # 2. 提取化合物 (返回的是 Tensor [B, L, H])
            d_feats = self.feature_extractor.drug_fea_extract_chemberta(compound_batch)
            # 将 batch 拆开存入 list
            all_drug_features.extend([d_feats[i] for i in range(d_feats.size(0))])
            # drug_features = self.feature_extractor.drug_fea_extract_chemberta(compound_batch)

            all_labels.append(label_batch.cpu())
        
        
        # 合并标签
        all_labels = torch.cat(all_labels, dim=0)

        # 保存为 dict。注意：protein 和 drug 此时是 List[Tensor]
        torch.save({
            "protein": all_protein_features,
            "drug": all_drug_features,
            "label": all_labels
        }, self.save_path)

        print(f"✅ 保存完成到{self.save_path}！共 {len(all_labels)} 条数据。")
        


class EncodedDTIDataset(Dataset):
    '''
    输入：
        encoded_path的内容是三元组:(protein_encoder_tensor, compound_encoder_tensor, label_tensor)
    
    输出：
        dataloader形式
    '''
    
    def __init__(self, encoded_path):
        data = torch.load(encoded_path)
        self.protein = data["protein"]   # (N, T, D) # List of Tensor
        self.drug = data["drug"]         # (N, T, D) # List of Tensor
        self.label = data["label"]      # Tensor

    def __len__(self):
        return self.label.size(0)

    def __getitem__(self, idx):

        return self.drug[idx], self.protein[idx], self.label[idx]



# --- 调用演示 ---

def data_preEncoder(hit_path, hit_encoder_path, drugbank_path,  drugbank_encoder_path,bs=32):
    """
    读取序列数据文件, 对文件进行encoder预处理

    hit_path、drugbank_path : 文本路径
    hit_encoder_path、drugbank_encoder_path : encoder向量的路径
    """

    # 1. 预处理
    # drugbank_path = Path("./data/dataset/drugbank/drugbank.csv")
    # hit_path = Path("./data/dataset/hit/hit.csv")

    # 序列dataloader
    hit_dataset = DTIDataset(hit_path)
    hit_dataloader = DataLoader(hit_dataset, batch_size=bs, shuffle=True, num_workers=0) # 序列的loader

    drugbank_dataset = DTIDataset(drugbank_path)
    drugbank_dataloader = DataLoader(drugbank_dataset, batch_size=bs, shuffle=True, num_workers=0)


    # encoder后的dataloader
    hit_encoder = SequencePreEncoder(hit_dataloader, hit_encoder_path)
    hit_encoder.encode_and_save()

    drugbank_encoder = SequencePreEncoder(drugbank_dataloader, drugbank_encoder_path)
    drugbank_encoder.encode_and_save()



