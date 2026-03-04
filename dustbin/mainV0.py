# import sys
# print("当前Python路径:", sys.executable)

from feature_extract import *
from multi_fusion import *
from moe import *
from feature_decoder import *
from interaction_pred import *
import pandas as pd

# import torch.utils.data.Dataset
# import torch.utils.data.DataLoader


# 1. 提取蛋白质和化合物的序列特征
"""
理论上来说,每个lst存储了多个蛋白质和化合物序列的特征
# 不同的化合物，他们的分词个数也不同 
"""

protein = "MYRPARVTSTSRFLNPYVVCFIVVAGVVILAVTIALLVYFLAFDQKSYFYRSSFQLLNVEYNSQLNSPATQEYRTLSGRIESLITKTFKESNLRNQFIRAHVAKLRQDGSGVRADVVMKFQFTRNNNGASMKSRIESVLRQMLNNSGNLEINPSTEITSLTDQAAANWLINECGAGPDLITLSEQRILGGTEAEEGSWPWQVSLRLNNAHHCGGSLINNMWILTAAHCFRSNSNPRDWIATSGISTTFPKLRMRVRNILIHNNYKSATHENDIALVRLENSVTFTKDIHSVCLPAATQNIPPGSTAYVTGWGAQEYAGHTVPELRQGQVRIISNDVCNAPHSYNGAILSGMLCAGVPQGGVDACQGDSGGPLVQEDSRRLWFIVGIVSWGDQCGLPDKPGVYTRVTAYLDWIRQQTGI"
compound = "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O"

FeatureExtractor = FeatureExtractor()

protein_feature_lst1 = FeatureExtractor.pro_fea_extract(protein)     #
# compound_feature_lst1 = FeatureExtractor.drug_fea_extract(compound) # 1024的特征张量 fingerprint提取
compound_feature_lst1 = FeatureExtractor.drug_fea_extract_chemberta(compound) # 


print(protein_feature_lst1[0].shape)     # [416, 100]
print(compound_feature_lst1[0].shape)       # fingerprint [1024,];  chemberta [43,768]

'''
假装蛋白质提取了三种模态，化合物也提取了三种模态
三种模态的shape一样
'''
protein_feature_lst2 = protein_feature_lst1     
compound_feature_lst2 = compound_feature_lst1 

protein_feature_lst3 = protein_feature_lst1     
compound_feature_lst3 = compound_feature_lst1  




# 2. 特征融合 —— 采用注意力机制

# 2.1  蛋白质特征融合
# 示例输入维度(超参数)
# pro_X = (416, 100) (token_len, d_model)
sequence_dim = 416
structure_dim = 416 # 256
knowledge_dim = 416 # 64
d_model = 100

# 创建模型
pro_fusion_model = CrossModalFusionModel(sequence_dim, structure_dim, knowledge_dim, d_model)

# 前向传播
protein_fused_output, pro_fusion_loss = pro_fusion_model(protein_feature_lst1[0], protein_feature_lst2[0], protein_feature_lst3[0])

print("==2. ============================")
print("protein_fused_output:", protein_fused_output.shape)  # 输出形状应为 (10, sequence_dim)
print("pro_fusion_loss :", pro_fusion_loss.item())  # 打印总损失值



## 2.2 化合物特征融合
# 示例输入维度(超参数)
# drug_X = (43, 768) (token_len, d_model)
sequence_dim = 43
structure_dim = 43 # 256
knowledge_dim = 43 # 64
d_model = 768
# 创建模型, 提取化合物融合特征
compound_fusion_model = CrossModalFusionModel(sequence_dim, structure_dim, knowledge_dim, d_model)
compound_fused_output, compound_fusion_loss = compound_fusion_model(compound_feature_lst1[0], compound_feature_lst2[0], compound_feature_lst3[0])

print("============================")
print("compound_fused_output Output:", compound_fused_output.shape)  # 输出形状应为 (10, sequence_dim)
print("compound_fusion_loss :", compound_fusion_loss.item())  # 打印总损失值


# protein_fused_output : (token_num, d_model) (416, 100)
# compound_fused_output : (token_num, d_model)  (43,768)




# 3. 专家编码器 : 不同蛋白用不同专家编码 
pro_moe_dim = 256
drug_moe_dim = 256
pro_fusion_dim = 100
drug_fusion_dim = 768

print("==3. ============================")
basic_pro_moe = BasicMOE(pro_fusion_dim, pro_moe_dim, 3)   # (feature_in, feature_out, expert_num)
protein_moe_output, protein_moe_loss = basic_pro_moe(protein_fused_output)
# print(protein_moe_output)
print("protein_moe_output.shape:", protein_moe_output.shape)  #(416,256) (batch, feature_out)
print("protein_moe_loss:", protein_moe_loss)


print("============================")
basic_compound_moe = BasicMOE(drug_fusion_dim, drug_moe_dim, 3)   # (feature_in=1024, feature_out=512, expert_num)
compound_moe_output, compound_moe_loss = basic_compound_moe(compound_fused_output)
# print(compound_moe_output)
print("compound_moe_output.shape:", compound_moe_output.shape)  #(43,256)  
print("compound_moe_loss:", compound_moe_loss)




# 4. 药物重建损失
# FeatureDecoder = FeatureDecoder(drug_moe_dim, pro_moe_dim, )


# 5. 蛋白质、化合物相互作用预测
print("==5. ============================")

pro_feat_dim = 256
drug_feat_dim = 256
emb_dim = 128

predictor = Predictor(pro_feat_dim, drug_feat_dim, emb_dim, "MF")
score = predictor.forward(protein_moe_output, compound_moe_output)
print("pred score is : ", score)




"""
"""
