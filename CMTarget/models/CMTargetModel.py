import torch
import torch.nn as nn
import os
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale_factor, dropout=0.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.scale_factor, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(scale_factor=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 默认对最后一个维度初始化

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(SelfAttentionPooling, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        scores = torch.tanh(self.linear1(x))
        scores = self.linear2(scores)
        scores = scores.squeeze(-1)

        attn_weights = F.softmax(scores, dim=-1)

        attn_weights = attn_weights.unsqueeze(-1)
        pooled = torch.sum(x * attn_weights, dim=-2)
        return pooled, attn_weights

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()[:2]
    batch_size, len_k = seq_k.size()[:2]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.layer_norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim * emb_dim))

    def forward(self, emb_fea, seq_index):
        mask = (seq_index == 0).float()
        event_K = self.event_K(emb_fea)
        t = event_K - torch.unsqueeze(mask, 2) * 1e8
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea, 1)
        his_fea = self.layer_norm(his_fea)
        output = self.decoder(his_fea)
        return output.squeeze(1)

class DrugEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.drug_mapping = torch.nn.Sequential(torch.nn.Linear(34, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, emb_dim))
        self.drug_self_attn = MultiHeadAttention(8, self.emb_dim, 64, 64, dropout=0.1)
        self.layer_norm = nn.LayerNorm(self.emb_dim, eps=1e-6)
        self.pooling = SelfAttentionPooling(self.emb_dim)

    def GCN(self, node_feat, adj_matrix):
        # node_feat: [batch_size, max_atom_num, atom_fea]
        # adj_matrix: [batch_size, max_atom_num, max_atom_num]

        conv_feature = torch.matmul(adj_matrix, node_feat)

        return conv_feature

    def forward(self, drug_feature):
        item_feature, adj_matrix = drug_feature
        # item_feature: [batch_size, max_atom_num, atom_fea]
        # adj_matrix: [batch_size, max_atom_num, max_atom_num]

        # 1. GCN, item_conv_feature: [batch_size, max_atom_num, atom_fea]
        item_conv_feature = self.GCN(item_feature, adj_matrix)

        # 2. Mapping To High Dim, mapping_feature: [batch_size, max_atom_num, emb_dim]
        item_mapping_feature = self.drug_mapping(item_conv_feature)

        # 3. Attention, item_att_feature: [batch_size, max_atom_num, emb_dim]
        # mask_feature = torch.sum(item_feature, dim=-1)
        # item_mask = get_attn_pad_mask(mask_feature, mask_feature)
        shape = item_mapping_feature.shape
        item_mapping_feature = item_mapping_feature.view(-1, shape[-2], shape[-1])
        item_attn_feature, _ = self.drug_self_attn(item_mapping_feature, item_mapping_feature, item_mapping_feature, mask=None)
        item_attn_feature = item_attn_feature.reshape(shape)
        item_attn_feature, _ = self.pooling(item_attn_feature)

        return item_attn_feature

class ProteinEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.protein_mapping = torch.nn.Sequential(torch.nn.Linear(1024, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.protein_attn_mapping = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
        self.layer_norm = nn.LayerNorm(self.emb_dim, eps=1e-6)

    def forward(self, protein_feature):
        # 1. Protein Mapping To Low Dim, user_mapping_feature: [batch_size, emb_dim]
        user_mapping_feature = self.protein_mapping(protein_feature)

        # 2. Compute Attention, user_attn: [batch_size, emb_dim]
        user_attn_score = self.protein_attn_mapping(user_mapping_feature)
        user_attn = torch.softmax(user_attn_score, dim=-1)

        # 3. Compute Protein Attention Feature, user_attn_feature: [batch_size, emb_dim]
        user_attn_feature = user_attn * user_mapping_feature
        user_attn_feature = user_attn_feature + user_mapping_feature
        user_attn_feature = self.layer_norm(user_attn_feature)

        return user_attn_feature

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
        reaction_result = user_embedding * item_embedding  # [batch_size, emb_dim]
        reaction_result = self.linear(reaction_result)
        output = torch.sum(reaction_result, dim=1)
        output = torch.sigmoid(output)
        return output

class Cosine(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

    def forward(self, user_embedding, item_embedding):
        output = torch.cosine_similarity(user_embedding, item_embedding, dim=1)
        output = torch.sigmoid(output)
        return output

class CMTargetModel(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim, score, timestamp):
        super().__init__()
        self.stamp = timestamp
        self.emb_dim = emb_dim # 512
        self.meta_dim = meta_dim
        self.drug_encoder = DrugEncoder(self.emb_dim)
        self.protein_encoder = ProteinEncoder(self.emb_dim)
        self.protein_bert_mapping = torch.nn.Linear(in_features=1024, out_features=self.emb_dim)
        self.protein_kg_mapping = torch.nn.Linear(in_features=1500, out_features=self.emb_dim)
        self.drug_bert_mapping = torch.nn.Linear(in_features=768, out_features=self.emb_dim)
        self.drug_kg_mapping = torch.nn.Linear(in_features=1500, out_features=self.emb_dim)
        self.fcn = nn.Linear(in_features=self.emb_dim, out_features=1)
        self.meta_net = MetaNet(emb_dim, meta_dim)
        self.layer_norm = nn.LayerNorm(self.emb_dim, eps=1e-6)
        self.relu = nn.ReLU()
        self.user_pooling = SelfAttentionPooling(self.emb_dim)
        self.item_pooling = SelfAttentionPooling(self.emb_dim)
        if score == 'MF':
            self.score = MF(emb_dim)
        elif score == 'GMF':
            self.score = GMF(emb_dim)
        elif score == 'Cosine':
            self.score = Cosine(emb_dim)

    def GCN(self, node_feat, adj_matrix):
        # node_feat: [batch_size, max_atom_num, atom_fea]
        # adj_matrix: [batch_size, max_atom_num, max_atom_num]

        conv_feature = torch.matmul(adj_matrix, node_feat)

        return conv_feature

    def forward(self, x):
        item_feature, user_feature, his_item_feature = x

        atom_feature, adj_matrix, drug_bert_feature, drug_kg_feature = item_feature
        protein_bert_feature, protein_kg_featre = user_feature

        # 得到特征表示
        item_attn_feature = self.drug_encoder((atom_feature, adj_matrix))
        user_attn_feature = self.protein_encoder(protein_bert_feature)
        his_atom_feature = torch.stack([feature[0] for feature in his_item_feature])
        his_adj_matrix = torch.stack([feature[1] for feature in his_item_feature])
        his_atom_feature = his_atom_feature.transpose(0, 1)
        his_adj_matrix = his_adj_matrix.transpose(0, 1)

        user_bert_feature = self.protein_bert_mapping(protein_bert_feature)
        user_kg_feature = self.protein_kg_mapping(protein_kg_featre)

        user_bert_feature = self.layer_norm(user_bert_feature)
        user_kg_feature = self.layer_norm(user_kg_feature)

        # 1. 
        user_fusion_feature = torch.stack([user_bert_feature, user_kg_feature], dim=1) 

        item_attn_feature = self.layer_norm(item_attn_feature)
        item_bert_feature = self.drug_bert_mapping(drug_bert_feature)
        item_kg_feature = self.drug_kg_mapping(drug_kg_feature)

        item_bert_feature = self.layer_norm(item_bert_feature)
        item_kg_feature = self.layer_norm(item_kg_feature)

        # 2. 将特征映射到同样的维度
        item_fusion_feature = torch.stack([item_attn_feature, item_bert_feature, item_kg_feature], dim=1)

        user_pool_feature, _ = self.user_pooling(user_fusion_feature)
        item_pool_feature, _ = self.item_pooling(item_fusion_feature)

        # 3. 预测得分
        output = self.score(user_pool_feature, item_pool_feature)

        return output

    def save_model(self):
        model_path = os.path.join("checkpoints", f"{self.stamp}_{'AttFusion'}.pt")
        torch.save(self.state_dict(), model_path)

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))