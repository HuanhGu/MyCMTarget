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
    proteins, drugs, labels, smiles, sequences = zip(*batch)
    # drugs, proteins, labels = zip(*batch)
    # 将一个 batch 内的序列 pad 到当前 batch 的最大长度
    drugs_pad = pad_sequence(drugs, batch_first=True)
    proteins_pad = pad_sequence(proteins, batch_first=True)
    labels_tensor = torch.stack(labels)
    return drugs_pad, proteins_pad, labels_tensor, smiles, sequences


class DTIDataset(Dataset):
    '''
    输入： 
        df: 包含 (compound, protein, label) 的 pandas DataFrame
    输出：
        dataloader形式 
    '''
    def __init__(self, df):
        # 允许直接接收 DataFrame 对象
        self.data = df.reset_index(drop=True) # 重置索引，防止 slice 后索引不连续导致错误

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 获取数据
        compound = row["compound"]
        protein = row["protein"]
        # 确保 label 是 float32 格式，适合 BCELoss 等损失函数
        label = torch.tensor(row["label"], dtype=torch.float32)

        return compound, protein, label


class SequenceEncoder:
    '''
    对蛋白质和化合物序列进行编码
    输入:
        dataloader : (compound_smiles, protein_sequence, label) # 注意顺序
    '''
    def __init__(self, dataloader, feature_extractor):
        self.dataloader = dataloader
        # 建议外部传入 feature_extractor 实例，避免内部重复初始化消耗显存
        self.feature_extractor = feature_extractor

    def encode_and_save(self, encoder_path, shuffle_path):
        # 一、初始化容器
        all_protein_features = []
        all_drug_features = []
        all_labels = []
        data_list = [] 
        
        print(f"🚀 开始预编码特征 (设备: {self.feature_extractor.device})...")

        # 使用 no_grad 节省显存，编码阶段不需要梯度
        with torch.no_grad():
            for compound_batch, protein_batch, label_batch in tqdm(self.dataloader):
                
                # --- 任务1：提取特征 (Tensor) --- 
                # 1.1 提取蛋白质
                p_feats = self.feature_extractor.pro_fea_extract(protein_batch)
                # 确保 p_feats 转移到 CPU，否则会撑爆显存
                all_protein_features.extend([p.cpu() for p in p_feats])
                
                # 1.2 提取化合物
                d_feats = self.feature_extractor.drug_fea_extract_chemberta(compound_batch)
                all_drug_features.extend([d.cpu() for d in d_feats])
                
                # 1.3 标签
                all_labels.append(label_batch.cpu())

                # --- 任务2：保存原始序列 (CSV) ---
                labels_np = label_batch.cpu().numpy()
                for i in range(len(compound_batch)):
                    data_list.append({
                        "compound": compound_batch[i],
                        "protein": protein_batch[i],
                        "label": labels_np[i]
                    })
        
        # --- 任务1 保存 (.pt 文件) ---  (每个特征向量未对齐)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        torch.save({
            "protein": all_protein_features,
            "drug": all_drug_features,
            "label": all_labels_tensor
        }, encoder_path) # 修正：使用传入的参数名
        print(f"✅ 特征保存完成：{encoder_path} | 共 {len(all_labels_tensor)} 条")

        # --- 任务2 保存 (.csv 文件) ---
        df = pd.DataFrame(data_list)
        df.to_csv(shuffle_path, index=False, encoding='utf-8') # 修正：使用传入的参数名
        print(f"✅ 序列保存完成：{shuffle_path} | 共 {len(df)} 条")

class PrecomputedCombinedDataset(Dataset):
    def __init__(self, encoder_path, shuffle_csv_path):
        # 加载之前保存的 .pt 特征字典
        self.features = torch.load(encoder_path, map_location="cpu")
        # 加载之前保存的已打乱序列 .csv
        self.df = pd.read_csv(shuffle_csv_path)
        
        # 确保两者的长度一致
        assert len(self.df) == len(self.features["label"]), "特征和序列数量不匹配！"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. 获取特征数据
        protein_feat = self.features["protein"][idx]
        drug_feat = self.features["drug"][idx]
        label = self.features["label"][idx]
        
        # 2. 获取原始序列数据 (smiles, sequence)
        smiles = self.df.iloc[idx]['compound']
        sequence = self.df.iloc[idx]['protein']
        
        # 同时返回，保证在一个 Batch 里绝对对齐
        return protein_feat, drug_feat, label, smiles, sequence
    