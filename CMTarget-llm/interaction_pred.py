import torch
from torch import nn
import torch.nn.functional as F

from model import *
from data_process import *



class GMF(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fcn = nn.Linear(in_features=emb_dim, out_features=1)

    def forward(self, user_embedding, item_embedding):
        reaction_result = user_embedding * item_embedding  # [batch_size, max_atom_num, emb_dim]
        output = self.fcn(reaction_result).squeeze(1)
        output = torch.sigmoid(output)
        return output

class MF(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(in_features=emb_dim, out_features=1)

    def forward(self, user_embedding, item_embedding):
        reaction_result = user_embedding * item_embedding  # [batch_size, emb_dim][2,256]
        reaction_result = self.linear(reaction_result) # [2,256] →  [2,1]
        # output = torch.sum(reaction_result, dim=1)
        output = torch.sum(reaction_result, dim=-1) # 修改,当batch=1时，dim就是-1[2]
        output = torch.sigmoid(output)
        return output

class Cosine(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

    def forward(self, user_embedding, item_embedding):
        output = torch.cosine_similarity(user_embedding, item_embedding, dim=1)
        output = torch.sigmoid(output)
        return output



class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(SelfAttentionPooling, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x): # 输入x维度:[token_num, dim]
        scores = torch.tanh(self.linear1(x))    # (416,256) → (416,128)
        scores = self.linear2(scores)           # (416,128) → (416,1)
        scores = scores.squeeze(-1)             # (416,1) → (416)

        attn_weights = F.softmax(scores, dim=-1)   # [416]

        attn_weights = attn_weights.unsqueeze(-1) # (416) → (416, 1)
        pooled = torch.sum(x * attn_weights, dim=-2)    # [256]
        return pooled, attn_weights
    


class Predictor():
    '''
    预测输入蛋白质和化合物的交互
    

    '''
    
    def __init__(self, pro_dim, drug_dim, emb_dim, score):
        self.pro_dim = pro_dim
        self.drug_dim = drug_dim
        self.emb_dim = emb_dim

        self.fea_dim = pro_dim

        self.user_pooling = SelfAttentionPooling(self.pro_dim, emb_dim)
        self.item_pooling = SelfAttentionPooling(self.drug_dim, emb_dim)


        if score == 'MF':
            self.score = MF(self.fea_dim)
        elif score == 'GMF':
            self.score = GMF(self.fea_dim)
        elif score == 'Cosine':
            self.score = Cosine(self.fea_dim)

    def get_model():
        pass



    
    def forward(self, pro_feat, drug_feat):

        # 1. 将输入映射到同一维度
        prot_pool_feature, _ = self.user_pooling(pro_feat)  # [2,256]
        drug_pool_feature, _ = self.item_pooling(drug_feat) #[2,256]
        

        # 2. 预测打分
        output = self.score(prot_pool_feature, drug_pool_feature)
        
        return output