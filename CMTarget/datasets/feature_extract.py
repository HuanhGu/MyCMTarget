import pandas as pd
import warnings
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
import re
from rdkit.Chem import AllChem
from rdkit import RDLogger
import torch
from datasets.extract_drug_graph import *

warnings.filterwarnings('ignore')
rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)

class FeatureExtractor(object):
    def __init__(self, protein_mode, drug_mode, configs):
        self.protein_mode = protein_mode
        self.drug_mode = drug_mode
        self.configs = configs
        self.feature_dim = 1024
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # protein extract mode
        if self.protein_mode == 'bert':
            self.feature_dim = 1024
            # download model from https://huggingface.co/Rostlab/prot_bert/tree/main
            self.prot_bert_model = BertModel.from_pretrained("model_hub/Rostlab/prot_bert")
            self.prot_tokenizer = BertTokenizer.from_pretrained("model_hub/Rostlab/prot_bert", do_lower_case=False)
            self.prot_bert_model = self.prot_bert_model.to(self.device)
        elif self.protein_mode == 'bert_dict':
            pro_fea_df = pd.read_csv("data/feature/prot_bert_{}_{}_feature.csv".format(self.configs['source'], self.configs['target']))
            pro_fea_dict = {row[0]: np.array(row[1:]).astype(np.float32) for row in pro_fea_df.values}
            self.protein_feature_dict = pro_fea_dict
        elif self.protein_mode == 'fusion':
            pro_bert_fea_df = pd.read_csv("data/feature/prot_bert_{}_{}_feature.csv".format(self.configs['source'], self.configs['target']))
            pro_bert_fea_dict = {row[0]: np.array(row[1:]).astype(np.float32) for row in pro_bert_fea_df.values}
            pro_kg_fea_df = pd.read_csv("data/feature/prot_kg_{}_{}_feature.csv".format(self.configs['source'], self.configs['target']))
            pro_kg_fea_dict = {row[0]: np.array(row[1:]).astype(np.float32) for row in pro_kg_fea_df.values}
            self.protein_bert_feature_dict = pro_bert_fea_dict
            self.protein_kg_feature_dict = pro_kg_fea_dict

        # drug extract mode
        if self.drug_mode == 'bert':
            # download model from https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1
            self.drug_bert_model = AutoModel.from_pretrained("model_hub/seyonec/ChemBERTa-zinc-base-v1")
            self.drug_tokenizer = AutoTokenizer.from_pretrained("model_hub/seyonec/ChemBERTa-zinc-base-v1")
            self.drug_bert_model = self.drug_bert_model.to(self.device)
        elif self.drug_mode == 'bert_dict':
            drug_fea_df = pd.read_csv("data/feature/drug_bert_{}_{}_feature.csv".format(self.configs['source'], self.configs['target']))
            drug_fea_dict = {row[0]: np.array(row[1:]).astype(np.float32) for row in drug_fea_df.values}
            self.drug_feature_dict = drug_fea_dict
        elif self.drug_mode == 'fusion':
            drug_bert_fea_df = pd.read_csv("data/feature/drug_bert_{}_{}_feature.csv".format(self.configs['source'], self.configs['target']))
            drug_bert_fea_dict = {row[0]: np.array(row[1:]).astype(np.float32) for row in drug_bert_fea_df.values}
            drug_kg_fea_df = pd.read_csv("data/feature/drug_kg_{}_{}_feature.csv".format(self.configs['source'], self.configs['target']))
            drug_kg_fea_dict = {row[0]: np.array(row[1:]).astype(np.float32) for row in drug_kg_fea_df.values}
            self.drug_bert_feature_dict = drug_bert_fea_dict
            self.drug_kg_feature_dict = drug_kg_fea_dict

    # bert method to extract feature
    def protein_bert_extract(self, protein_sequence):
        protein_sequence = re.sub(r"[UZOB]", "X", protein_sequence)
        protein_sequence = ' '.join(protein_sequence)
        encoded_input = self.prot_tokenizer(protein_sequence, return_tensors='pt', padding='max_length', truncation=True, max_length=1024)
        encoded_input = encoded_input.to(self.device)
        with torch.no_grad():
            output = self.prot_bert_model(**encoded_input)
        feature = output[0].data.cpu().numpy()
        feature = feature.mean(axis=1).reshape(-1)

        return feature
    def drug_bert_extract(self, drug_sequence):
        inputs = self.drug_tokenizer(drug_sequence, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        inputs = inputs.to("cuda")
        # 模型前向传播计算
        with torch.no_grad():
            outputs = self.drug_bert_model(**inputs)

        # 提取 CLS token 对应的分子嵌入
        molecule_embedding = outputs.last_hidden_state[:, 0, :]
        molecule_embedding = molecule_embedding.cpu().numpy().reshape(-1)
        return molecule_embedding

    # generate drug feature with MorganFingerprint
    def fingerprint_extract(self, drug):
        if Chem.MolFromSmiles(drug):
            mol = Chem.MolFromSmiles(drug)
            radius = 2
            nBits = self.feature_dim
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            fingerprint_feature = np.array(fingerprint, dtype=np.float32)
        else:
            # print(str(drug))
            # print("Above smile transforms to fingerprint error!!!")
            # print("Please remove!")
            fingerprint_feature = np.zeros(self.feature_dim, dtype=np.float32)

        return fingerprint_feature

    def graph_extract(self, drug):
        atom_feat, adj_matrix = drug_graph_extract(drug)
        return atom_feat, adj_matrix

    def extract_protein(self, protein_sequence):
        protein_feature = None
        if self.protein_mode == 'bert':
            protein_feature = self.protein_bert_extract(protein_sequence)
            return protein_feature
        elif self.protein_mode == 'bert_dict':
            protein_feature = self.protein_feature_dict[protein_sequence]
            return protein_feature
        elif self.protein_mode == 'fusion':
            protein_bert_feature = self.protein_bert_feature_dict[protein_sequence]
            protein_kg_feature = self.protein_kg_feature_dict[protein_sequence]
            return (protein_bert_feature, protein_kg_feature)

    def extract_drug(self, drug_sequence):
        drug_feature = None
        if self.drug_mode == 'fingerprint':
            drug_feature = self.fingerprint_extract(drug_sequence)
            return drug_feature
        elif self.drug_mode == 'graph':
            # atom_feat: [100, 34]
            # adj_matrix: [100, 100]
            atom_feat, adj_matrix = self.graph_extract(drug_sequence)
            return (atom_feat, adj_matrix)
        elif self.drug_mode == 'fusion':
            # atom_feat: [100, 34]
            # adj_matrix: [100, 100]
            atom_feat, adj_matrix = self.graph_extract(drug_sequence)
            if drug_sequence == '':
                drug_bert_feature = np.zeros(768, dtype=np.float32)
                drug_kg_feature = np.zeros(1500, dtype=np.float32)
            else:
                drug_bert_feature = self.drug_bert_feature_dict[drug_sequence]
                drug_kg_feature = self.drug_kg_feature_dict[drug_sequence]

            return (atom_feat, adj_matrix, drug_bert_feature, drug_kg_feature)