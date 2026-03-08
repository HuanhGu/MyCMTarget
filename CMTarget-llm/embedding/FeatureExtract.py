import warnings
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
from rdkit.Chem import AllChem
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from rdkit import Chem
from gensim.models import Word2Vec


class FeatureExtractor(object):
    '''
    对蛋白质和化合物序列 分词+提取tokens特征


    ## 输出 : [batch_size, token_len, token_feature_dim]
    '''
    def __init__(self):
        # self.configs = configs
        self.feature_dim = 1024
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 预加载模型，避免在循环中重复加载
        print("Loading Word2Vec model...")
        self.w2v_model = Word2Vec.load("./embedding/word2vec_30.model")
        
        print("Loading ChemBERTa model...")
        self.drug_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.drug_model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(self.device)
        self.drug_model.eval()


    # TransformerCPI-Kinase：把蛋白质序列分成若干个氨基酸 
    def seq_to_kmers(self, seq, k=3):
        """ Divide a string into a list of kmers strings.

        Parameters:
            seq (string)
            k (int), default 3
        Returns:
            List containing a list of kmers.
        """
        N = len(seq)
        return [seq[i:i+k] for i in range(N - k + 1)]
    

    def get_protein_embedding(self, model,protein):
        """get protein embedding,infer a list of 3-mers to (num_word,100) matrix"""
    
        vec = np.zeros((len(protein), 100))
        i = 0
        for word in protein:
            vec[i, ] = model.wv[word]
            i += 1
        return vec

    def pro_fea_extract(self, pro_sequence):
        '''
        提取一个batch蛋白质序列的特征编码tensor

        输入：
            蛋白质序列list : [batch_size, ]个sequence
        输出：
            蛋白质序列的张量嵌入list : [batch_size, token_num, Hidden_Size ]
        '''
        proteins = []
        for seq in pro_sequence:
            kmers = self.seq_to_kmers(seq)

            # 查表操作在 CPU 上完成
            vec = np.array([self.w2v_model.wv[w] for w in kmers if w in self.w2v_model.wv])
            proteins.append(torch.FloatTensor(vec))

        # 返回 list，由外部处理 padding 逻辑
        return proteins
    
        # 原始我的版本
        # for seq in pro_sequence:
            # [提取 1 个蛋白质序列的特征]
            # model = Word2Vec.load("./embedding/word2vec_30.model")
            # protein_embedding = self.get_protein_embedding(self.w2v_model, self.seq_to_kmers(seq))
            # protein = torch.FloatTensor(protein_embedding)
            # proteins.append(protein)

        # proteins = pad_sequence(proteins, batch_first=True)
        # return proteins




    # https://github.com/miservilla/ChemBERTa
    def drug_fea_extract_chemberta(self, drug_sequence):
        """
        提取一个batch化合物序列的特征编码tensor

        输入：
            drug序列list : [batch_size, ]个 list of SMILES
        输出：
            drug序列的张量嵌入list : [batch_size, token_num, Hidden_Size]
        
        """
        
        inputs = self.drug_tokenizer(drug_sequence, return_tensors="pt", 
                                     padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.drug_model(**inputs)
        
        # 结果转回 CPU 释放显存
        return outputs.last_hidden_state.cpu()
    



    
        # 原始我的版本
        # model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1") # print(model)
        # tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        # inputs = tokenizer(drug_sequence, return_tensors="pt", padding=True, truncation=True )
    
        # with torch.no_grad():
        #     outputs = model(**inputs)

        # # 返回所有 token 的特征，形状为 [Batch_Size, Sequence_Length, Hidden_Size]
        # cls_embedding = outputs.last_hidden_state  #[2,43,768]
        
        # cls_embedding = pad_sequence(cls_embedding, batch_first=True)
        # return cls_embedding



    # CMTarget：提取化合物序列的特征
    # generate drug feature with MorganFingerprint
    def drug_fea_extract(self, drug_sequence):  
        drugs = []

        # 提取1个化合物序列的特征
        if Chem.MolFromSmiles(drug_sequence):
            mol = Chem.MolFromSmiles(drug_sequence)
            radius = 2
            nBits = self.feature_dim
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            fingerprint_feature = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0)
        else:
            # print(str(drug))
            # print("Above smile transforms to fingerprint error!!!")
            # print("Please remove!")
            fingerprint_feature = torch.zeros(self.feature_dim, dtype=torch.float32).unsqueeze(0)

        drugs.append(fingerprint_feature)

        
        return drugs
        # drug_feature_lst1 = FeatureExtractor.drug_fea_extract(drug) # fingerprint提取 [1024,];


    


