from torch.utils.data import Dataset
from datasets.feature_extract import *

class CMTargetDataset(Dataset):
    def __init__(self, interactions, extractor):
        self.extractor = extractor
        self.user_data = interactions['uid'].values
        self.item_data = interactions['iid'].values
        self.y_data = interactions["y"].values

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        uid = self.user_data[idx]
        iid = self.item_data[idx]
        y = self.y_data[idx]

        user_feature = self.extractor.extract_protein(uid)
        item_feature, adj_matrix = self.extractor.extract_drug(iid)

        return (item_feature, adj_matrix), user_feature, y

class AttFusionMetaDataset(Dataset):
    def __init__(self, interactions, extractor, his_len=5):
        self.extractor = extractor
        self.user_data = interactions['uid'].values
        self.item_data = interactions['iid'].values
        self.his_data = interactions["pos_seq"].values
        self.y_data = interactions["y"].values
        self.his_len = his_len

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        uid = self.user_data[idx]
        iid = self.item_data[idx]
        his = self.his_data[idx]
        y = self.y_data[idx]

        user_feature = self.extractor.extract_protein(uid)
        item_feature = self.extractor.extract_drug(iid)

        his_item_feature_list = []
        for i in range(self.his_len):
            his_item_feature = self.extractor.extract_drug(his[i])
            his_item_feature_list.append(his_item_feature)

        return item_feature, user_feature, his_item_feature_list, y