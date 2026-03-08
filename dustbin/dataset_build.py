import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from embedding.FeatureExtract import *
from torch import nn
from tqdm import tqdm


class DTIDataset(Dataset):
    '''
    2. 构建dataset, 读取数据集文件(蛋白质序列、化合物序列、标签)

    输入： 
        file_path: 文件位置
    输出：
        compound: 化合物数据
        protein:蛋白质列
        label:标签列
    
    '''
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        # self.feature_extractor = FeatureExtractor()
        # print(self.data.columns)    #'compound,protein,label'
        # print("Columns:", self.data.columns)
        # print(self.data.head())
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        compound = row["compound"]
        protein = row["protein"]
        label = torch.tensor(row["label"], dtype=torch.float32)
        
        # protein_feat = self.feature_extractor.pro_fea_extract(protein)
        # compound_feat = self.feature_extractor.drug_fea_extract_chemberta(compound)
        
        return compound, protein, label



class SequencePreEncoder:
    '''
    对蛋白质和化合物序列进行multi modal encoder编码

    输入:
        dataloader : 蛋白质序列, 化合物序列, label
        savepath : 预编码完毕后的encoder保存路径

        protein : 蛋白质序列list [batch_size, ]个蛋白质序列
        compound : 化合物序列list [batch_size, ]个化合物序列

    输出:
        protein_encoder_modals:蛋白质的三种模态编码 [3, batch_size, token_num, token_dim]
        drug_encoder_modals:化合物的三种模态编码    [3, batch_size, token_num, token_dim]
    
    为什么我要把这一部分单独摘出来？
        因为有了序列后, encoder部分可以提前处理, 这样节约训练的时间。

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

        print("begin encoder —— every batch:")
        for compound_batch, protein_batch, label_batch in tqdm(self.dataloader):

            protein_features = self.feature_extractor.pro_fea_extract(protein_batch)
            drug_features = self.feature_extractor.drug_fea_extract_chemberta(compound_batch)

            # 构造三模态
            # print(type(protein_features))
            protein_modals = torch.stack(
                [protein_features, protein_features, protein_features],
                dim=0
            )  # (3, B, T, D)  [3,2,416,100]

            drug_modals = torch.stack(
                [drug_features, drug_features, drug_features],
                dim=0
            )   # [3,2,43,768]

            print("the protein_modals.shape is :{}, the drug_modals.shape is :{}".format(protein_modals.shape, drug_modals.shape))
            # list(tensor1, tensor2..)
            all_protein_features.append(protein_modals)
            all_drug_features.append(drug_modals)
            all_labels.append(label_batch)


            # 拼接所有 batch
            all_protein_features = torch.cat(all_protein_features, dim=1)   # 变为tensor类型
            all_drug_features = torch.cat(all_drug_features, dim=1)
            all_labels = torch.cat(all_labels, dim=0)
        
        
        torch.save({
            "protein": all_protein_features,
            "drug": all_drug_features,
            "label": all_labels
        }, self.save_path)

        print("✅ 编码完成并保存:", self.save_path)
        


class EncodedDTIDataset(Dataset):
    '''
    输出：
        dataloader形式,
        [3, batch_size,]条三元组数据,
        三元组内容:(protein_multiModal_encoder_tensor, compound_multiModal_encoder_tensor, label_tensor)
        encoder_tensor: [3, batch_size, token_num, token_dim]
        
    '''
    
    def __init__(self, encoded_path):
        data = torch.load(encoded_path)

        self.protein = data["protein"]   # (3, N, T, D)
        self.drug = data["drug"]         # (3, N, T, D)
        self.label = data["label"]

    def __len__(self):
        return self.label.size(0)

    def __getitem__(self, idx):
        return (
            self.protein[:, idx],   # (3, T, D)
            self.drug[:, idx],
            self.label[idx]
        )




import torch
from torch.utils.data import DataLoader

# 假设你的类都定义在相应的文件中
# from your_module import DTIDataset, SequencePreEncoder, EncodedDTIDataset

def main():
    # --- 阶段 1: 原始数据准备 ---
    raw_csv_path = "data/dti_raw.csv"
    encoded_file_path = "data/encoded_features.pt"
    
    # 实例化原始读取器
    raw_dataset = DTIDataset(raw_csv_path)
    # 这里 batch_size 可以设大一点，加快编码速度
    raw_loader = DataLoader(raw_dataset, batch_size=32, shuffle=False)

    # --- 阶段 2: 预编码并保存到硬盘 ---
    # 这一步只需运行一次，后续实验可以直接跳过
    print("开始预编码特征...")
    pre_encoder = SequencePreEncoder(raw_loader, encoded_file_path)
    pre_encoder.encode_and_save()

    # --- 阶段 3: 训练时调用 ---
    # 直接加载已经转为 Tensor 的特征文件
    print("加载预编码后的数据集...")
    train_dataset = EncodedDTIDataset(encoded_file_path)
    
    # 这里的 loader 吐出的就是 (3, T, D) 的特征向量了
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # --- 测试调用 ---
    for p_feat, d_feat, label in train_loader:
        print(f"蛋白质模态维度: {p_feat.shape}") # Expected: [batch, 3, T, D]
        print(f"药物模态维度: {d_feat.shape}")    # Expected: [batch, 3, T, D]
        print(f"标签维度: {label.shape}")
        break

if __name__ == "__main__":
    main()


    

''' 3. 构建 dataloader 

# 文件路径
drugbank_path = "./data/dataset/drugbank/drugbank.csv"
# drugbank_data = pd.read_csv(drugbank_path, sep="\t")

hit_path = "./data/dataset/hit/hit.csv"
# hit_data = pd.read_csv(hit_path, sep="\t")




hit_dataset = DTIDataset(hit_path)

hit_dataloader = DataLoader(
    hit_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)


# for compound_batch, protein_batch, label_batch in hit_dataloader:
#     print(len(compound_batch))   # 32
#     print(len(protein_batch))    # 32
#     print(label_batch.shape)     # torch.Size([32])

'''