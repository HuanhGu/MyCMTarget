import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDecoder(nn.Module):
    def __init__(self, drug_dim=1024, protein_dim=200, hidden_dim=256):
        super(FeatureDecoder, self).__init__()
        
        # 药物解码层
        # self.drug_decoder = nn.Linear(drug_dim, hidden_dim)
        self.drug_decoder = nn.Sequential(
                                            nn.Linear(drug_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim)
                                        )
        
        # 蛋白质解码层
        # self.protein_decoder = nn.Linear(protein_dim, hidden_dim)
        self.protein_decoder = nn.Sequential(
                                            nn.Linear(protein_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim)
                                        )
        
        # self.activation = nn.ReLU()

    def forward(self, drug_feat, protein_feat):
        """
        drug_feat: (batch_size, drug_dim)
        protein_feat: (batch_size, protein_dim)
        """
        
        drug_decoded = self.drug_decoder(drug_feat)
        protein_decoded = self.protein_decoder(protein_feat)
        
        return drug_decoded, protein_decoded
    

    def compute_reconstruction_loss(self, drug_feat, protein_feat, drug_decoded, protein_decoded):
        """
        计算重建损失（原始特征与解码特征之间的均方误差）。
        """
        
        # 使用均方误差（MSE）计算损失
        drug_loss = F.mse_loss(drug_decoded, drug_feat)
        protein_loss = F.mse_loss(protein_decoded, protein_feat)
        
        # 总损失是药物和蛋白质的重建损失之和
        total_loss = drug_loss + protein_loss
        
        return total_loss