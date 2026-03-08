import pandas as pd
from torch.utils.data import Dataset
import tqdm
from datasets.feature_extract import FeatureExtractor
from torch.utils.data import DataLoader

class ProteinDataset(Dataset):
    def __init__(self, configs, extractor):
        src_path = 'data/dataset/{}/{}.csv'.format(configs['source'], configs['source'])
        tgt_path = 'data/dataset/{}/{}.csv'.format(configs['target'], configs['target'])
        src_df = pd.read_csv(src_path)
        tgt_df = pd.read_csv(tgt_path)
        src_protein = set(src_df.protein.unique())
        tgt_protein = set(tgt_df.protein.unique())
        self.proteins = src_protein | tgt_protein
        self.proteins = list(self.proteins)
        self.extractor = extractor

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        feature = self.extractor.protein_bert_extract(self.proteins[idx])
        return self.proteins[idx], feature

class CompoundDataset(Dataset):
    def __init__(self, configs, extractor):
        src_path = 'data/dataset/{}/{}.csv'.format(configs['source'], configs['source'])
        tgt_path = 'data/dataset/{}/{}.csv'.format(configs['target'], configs['target'])
        src_df = pd.read_csv(src_path)
        tgt_df = pd.read_csv(tgt_path)
        src_compound = set(src_df.compound.unique())
        tgt_compound = set(tgt_df.compound.unique())
        self.compounds = src_compound | tgt_compound
        self.compounds = list(self.compounds)
        self.extractor = extractor

    def __len__(self):
        return len(self.compounds)

    def __getitem__(self, idx):
        feature = self.extractor.drug_bert_extract(self.compounds[idx])
        return self.compounds[idx], feature

def prot_bert_extract(configs):
    print("Pre extract proteins bert feature!")
    extractor = FeatureExtractor('bert', 'bert', configs)
    dataset = ProteinDataset(configs, extractor)
    protein_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    pro_df = pd.DataFrame()
    for proteins, feature in tqdm.tqdm(protein_dataloader, smoothing=0, mininterval=1.0):
        proteins = list(proteins)
        feature = feature.numpy()
        proteins_df = pd.DataFrame(proteins)
        feature_df = pd.DataFrame(feature)
        result_df = pd.concat([proteins_df, feature_df], axis=1)
        pro_df = pd.concat([pro_df, result_df], ignore_index=True)
    pro_df.to_csv(f'data/feature/prot_bert_{configs["source"]}_{configs["target"]}_feature.csv', sep=',', header=True, index=False)

    print(f'success extract feature to data/feature/prot_bert_{configs["source"]}_{configs["target"]}_feature.csv')

def drug_bert_extract(configs):
    print("Pre extract drugs bert feature!")
    extractor = FeatureExtractor('bert', 'bert', configs)
    dataset = CompoundDataset(configs, extractor)
    protein_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    pro_df = pd.DataFrame()
    for proteins, feature in tqdm.tqdm(protein_dataloader, smoothing=0, mininterval=1.0):
        proteins = list(proteins)
        feature = feature.numpy()
        proteins_df = pd.DataFrame(proteins)
        feature_df = pd.DataFrame(feature)
        result_df = pd.concat([proteins_df, feature_df], axis=1)
        pro_df = pd.concat([pro_df, result_df], ignore_index=True)
    pro_df.to_csv(f'data/feature/drug_bert_{configs["source"]}_{configs["target"]}_feature.csv', sep=',', header=True, index=False)

    print(f'success extract feature to data/feature/drug_bert_{configs["source"]}_{configs["target"]}_feature.csv')

