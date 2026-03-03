from torch import nn
import torch

from feature_extract import *
from multi_fusion import *
from moe import *
from feature_decoder import *
from interaction_pred import *

import pandas as pd
import torch.optim as optim
from tqdm import tqdm



class CMTargetModel(nn.Module):
    def __init__(self, configs):
        super(CMTargetModel, self).__init__()
        self.configs = configs

        #  2 特征融合的超参数
        self.pro_sequence_tklen = 416         # 蛋白质序列编码的token数目
        self.pro_structure_tklen = 416 # 256  # 蛋白质结构编码的token数目
        self.pro_knowledge_tklen = 416 # 64   # 蛋白质知识图谱提取的token数目 
        self.pro_token_dim = 100              # 每个token的维度


        self.drug_sequence_tklen = 43          # 化合物序列编码的token数目
        self.drug_structure_tklen = 43         # 化合物结构编码的token数目
        self.drug_knowledge_tklen = 43         # 化合物知识图谱提取的token数目 
        self.drug_token_dim = 768              # 每个token的维度


        # 3 专家编码超参数
        self.pro_fusion_dim = 100
        self.drug_fusion_dim = 768
        self.pro_moe_dim = 256
        self.drug_moe_dim = 256

        
        # 5. 预测超参数
        self.score_way = configs['score_way']   # 打分器选择
        self.score_emb_dim = 128                # 打分时的特征嵌入维度
        
        # 6. 模型可学习参数
        # === 创建 fusion 模型 =====
        self.pro_fusion_model = CrossModalFusionModel(self.pro_sequence_tklen, self.pro_structure_tklen, self.pro_knowledge_tklen, self.pro_token_dim)
        self.drug_fusion_model = CrossModalFusionModel(self.drug_sequence_tklen, self.drug_structure_tklen, self.drug_knowledge_tklen, self.drug_token_dim)
        

        # === 创建 基础专家 模型 ===
        self.basic_pro_moe = BasicMOE(self.pro_fusion_dim, self.pro_moe_dim, 3)   # (feature_in, feature_out, expert_num)[100,256]
        self.basic_drug_moe = BasicMOE(self.drug_fusion_dim, self.drug_moe_dim, 3)   # (feature_in, feature_out, expert_num)[768,256]
        
        # === 创建 预测 模型 ===
        self.predictor = Predictor(self.pro_moe_dim, self.drug_moe_dim, self.score_emb_dim, "MF")




    def forward(self, pro_encoder_modals, drug_encoder_modals):
        """
        model的前向传播

        输入:
            pro_encoder_modals: [3, batch_size, token_num, token_dim] 蛋白质三种模态的特征encoder tensor
            drug_encoder_modals:[3, batch_size, token_num, token_dim] 化合物三种模态的特征encoder tensor
        
        返回:
            pro_moe_output: 蛋白质序列经过特征提取、融合、moe编码后的特征向量, 
            drug_moe_output:化合物序列经过特征提取、融合、moe编码后的特征向量,   
            pro_emb_loss:   protein的特征嵌入损失
            drug_emb_loss:  drug的特征嵌入损失
        """

        # 2. 特征融合 —— 采用注意力机制
        # 2.1  蛋白质特征融合;2.2  化合物特征融合
        # pro_X = [3,2,501,100]  →  [2,501,100]    drug_X:[3,2,68,768] →  [2,68,768] 
        # 前向传播, 对比损失
        pro_fused_output, pro_fusion_loss = self.pro_fusion_model(pro_encoder_modals[0], pro_encoder_modals[1], pro_encoder_modals[2])
        drug_fused_output, drug_fusion_loss = self.drug_fusion_model(drug_encoder_modals[0], drug_encoder_modals[1], drug_encoder_modals[2])


        # 3. 专家编码器 : 不同蛋白和化合物的token用不同专家编码 
        # 专家编码输出, moe的负载均衡损失
        pro_moe_output, pro_moe_loss = self.basic_pro_moe(pro_fused_output) #in:[2,501,100] out:[2,501,256]
        drug_moe_output, drug_moe_loss = self.basic_drug_moe(drug_fused_output) #in:[2,68,78] out:[2,68,256]

        pro_emb_loss = pro_fusion_loss + pro_moe_loss
        drug_emb_loss = drug_fusion_loss + drug_moe_loss


        # 5. 预测 : 预测蛋白质和化合物的相互作用关系 in:[2,501,256]  [2,68,256]
        score = self.predictor.forward(pro_moe_output, drug_moe_output)

        return score, pro_emb_loss, drug_emb_loss




class CMTargetTrainer():
    """
    
    input:
        dataloader: (compound, protein, label), [3, batch_size, token_num, token_dim]

    
    """
    def __init__(self, configs, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.learning_rate = configs['learning_rate']
        self.epochs = configs['epochs']
        self.criterion = nn.BCELoss()  # 使用二分类交叉熵损失函数
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.feature_extractor = FeatureExtractor()

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for compound_batch, protein_batch, label_batch in tqdm(self.dataloader):        
                
                # 1. 对输入的数据进行encoder
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

                
                # 确保标签是一个列向量
                label = label_batch.unsqueeze(1)  
               
                # 清空梯度
                self.optimizer.zero_grad()

                # 前向传播：三种模态特征对齐融合+MoE编码 [3,2,501,100]  [3,2,68,768]
                outputs, pro_emb_loss, drug_emb_loss = self.model(protein_modals, drug_modals)
                
                # 计算损失  [2]  [2,1]
                label = label.squeeze(1) #label:[2,1] → [2]
                loss = self.criterion(outputs, label)
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # 计算准确率
                predicted = (outputs > 0.5).float()  # 将输出转换为0或1
                correct += (predicted == label).sum().item()
                total += label.size(0)

            avg_loss = running_loss / len(self.dataloader)
            accuracy = correct / total * 100
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")  





