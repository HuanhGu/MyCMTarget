import os.path
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class CDRDataPreprocessor():
    def __init__(self, src_domain, tgt_domain, data_dir, his_len):
        self.src_domain = src_domain
        self.tgt_domain = tgt_domain
        self.data_dir = os.path.join('data', data_dir)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.his_len = his_len

    def read_csv(self, domain):
        path = 'data/dataset/{}/{}.csv'.format(domain, domain)
        csv_df = pd.read_csv(path)
        csv_df.columns = ['iid', 'uid', 'y']
        csv_df = csv_df[['uid', 'iid', 'y']]
        return csv_df

    def get_history(self, data, uid_set):
        pos_seq_dict = {}
        for uid in tqdm(uid_set):
            pos = data[(data.uid == uid) & (data.y == 1)].iid.values.tolist()
            length = len(pos)
            if length > self.his_len:
                pos = pos[:self.his_len]
            elif length < self.his_len:
                while length < self.his_len:
                    pos.append('')
                    length += 1
            pos_seq_dict[uid] = pos

        return pos_seq_dict

    def split(self, src, tgt):
        src_users = set(src.uid.unique())
        tgt_users = set(tgt.uid.unique())
        # common users in src & tgt
        co_users = src_users & tgt_users
        # split train and test by user
        test_users = set(random.sample(co_users, round(0.2 * len(co_users))))
        pos_seq_dict = self.get_history(src, co_users)
        train_meta = tgt[tgt['uid'].isin(co_users - test_users)]
        test_meta = tgt[tgt['uid'].isin(test_users)]
        train_meta['pos_seq'] = train_meta.uid.map(pos_seq_dict)
        test_meta['pos_seq'] = test_meta.uid.map(pos_seq_dict)

        return train_meta, test_meta

    def run(self):
        src_data = self.read_csv(self.src_domain)
        tgt_data = self.read_csv(self.tgt_domain)
        train_meta, test_meta = self.split(src_data, tgt_data)

        print(f"train_meta:{len(train_meta)}, test_meta:{len(test_meta)}")

        train_meta.to_csv(f'{self.data_dir}/train_meta.csv', sep=',', header=["uid", "iid", "y", "pos_seq"], index=False)
        test_meta.to_csv(f'{self.data_dir}/test_meta.csv', sep=',', header=["uid", "iid", "y", "pos_seq"], index=False)

        print("data saved to {}".format(self.data_dir))