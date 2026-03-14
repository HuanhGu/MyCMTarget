

class PrecomputedDTIDataset(Dataset):
    def __init__(self, pt_path):
        print(f"📦 正在从内存加载预编码特征: {pt_path}")
        # 加载 torch.save 保存的 dict
        data = torch.load(pt_path)
        
        self.protein_feats = data["protein"] # List of Tensors
        self.drug_feats = data["drug"]       # List of Tensors
        self.labels = data["label"]          # Tensor [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 直接返回 Tensor，无需再通过模型编码
        return self.drug_feats[idx], self.protein_feats[idx], self.labels[idx]
    


    def get_dataloader(self, source_datapath):
        # 1. 定义获取单侧（Train或Test）数据的子函数
        def prepare_single_split(flag, df_split=None):
            encoder_path, shuffle_path = self.get_data_new_path(source_datapath, flag=flag)

            # 如果文件不存在且传入了数据，则执行编码并保存
            if not os.path.exists(encoder_path) and df_split is not None:
                print(f"🔍 编码 {flag} 数据中...")
                loader = DataLoader(DTIDataset(df_split), batch_size=self.batch_size, shuffle=True)
                
                encoder = SequenceEncoder(loader, self.feature_extractor)
                encoder.encode_and_save(encoder_path, shuffle_path)

            # 构造并返回两个 Loader
            feat_loader = DataLoader(PrecomputedDTIDataset(encoder_path), batch_size=self.batch_size)
            seq_loader = DataLoader(DTIDataset(pd.read_csv(shuffle_path)), batch_size=self.batch_size, shuffle=True)
            
            return feat_loader, seq_loader

        # 2. 主逻辑：检查并处理数据分割
        train_path, _ = self.get_data_new_path(source_datapath, flag="train")
        train_df, test_df = None, None
        
        if not os.path.exists(train_path):
            print("🔍 初始化数据分割...")
            df = pd.read_csv(source_datapath)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, shuffle=True)

        # 3. 得到最终结果
        train_feat_l, train_seq_l = prepare_single_split("train", train_df)
        test_feat_l, test_seq_l = prepare_single_split("test", test_df)

        return train_feat_l, test_feat_l, train_seq_l, test_seq_l
    




    '''
    def get_dataloader(self, source_datapath):
        batch_size = self.batch_size
        train_encoder_path, train_shuffle_path = self.get_data_new_path(source_datapath, flag="train")
        test_encoder_path, test_shuffle_path = self.get_data_new_path(source_datapath, flag="test")

        # 检查是否已经存在预处理好的文件,如果没有，先保存
        if not os.path.exists(train_encoder_path):
            print("🔍 未发现特征文件，开始执行编码流程...")
            # 1. 读取csv文件，数据划分
            df = pd.read_csv(source_datapath)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, shuffle=True)
            
            # 2. 将csv序列数据加载为 train_loader、test_loader
            # [smiles, sequence, label]
            train_ds_smiles = DTIDataset(train_df)
            train_loader = DataLoader(train_ds_smiles, batch_size=batch_size, shuffle=True)

            test_ds_smiles = DTIDataset(test_df)
            test_loader = DataLoader(test_ds_smiles, batch_size=batch_size, shuffle=True)

            # 3. 对序列编码, 保存shuffle后的序列数据.csv, 
            #                   以及encoder之后的.pt  [tensor, tensor, label]
            train_encoder = SequenceEncoder(train_loader, self.feature_extractor)
            train_encoder.encode_and_save(train_encoder_path, train_shuffle_path)

            test_encoder = SequenceEncoder(test_loader, self.feature_extractor)
            test_encoder.encode_and_save(test_encoder_path, test_shuffle_path)

        
        # 获取feature_loader
        train_dataset = PrecomputedDTIDataset(train_encoder_path)
        train_encoder_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataset = PrecomputedDTIDataset(test_encoder_path)
        test_encoder_loader = DataLoader(test_dataset, batch_size=batch_size)
        # 获取sequence_loader
        train_df = pd.read_csv(train_shuffle_path)
        train_ds_smiles = DTIDataset(train_df)
        train_sequence_loader = DataLoader(train_ds_smiles, batch_size=batch_size, shuffle=True)
        test_df = pd.read_csv(test_shuffle_path)
        test_ds_smiles = DTIDataset(test_df)
        test_sequence_loader = DataLoader(test_ds_smiles, batch_size=batch_size, shuffle=True)

        return train_encoder_loader, test_encoder_loader,train_sequence_loader,test_sequence_loader
'''
    


        
class EncodedDTIDataset(Dataset):
    '''
    输入：
        encoded_path的内容是三元组:(protein_encoder_tensor, compound_encoder_tensor, label_tensor)
    
    输出：
        dataloader形式
    '''
    
    def __init__(self, encoded_path):
        data = torch.load(encoded_path)
        self.protein = data["protein"]   # (N, T, D) # List of Tensor
        self.drug = data["drug"]         # (N, T, D) # List of Tensor
        self.label = data["label"]      # Tensor

    def __len__(self):
        return self.label.size(0)

    def __getitem__(self, idx):

        return self.drug[idx], self.protein[idx], self.label[idx]



        
        # model = self.get_model()
        # model = self.model
        # criterion = self.criterion
        # train_dataloader = self.train_dataloader
        # evl_dataloader = self.evl_dataloader
        # optimizer = self.optimizer