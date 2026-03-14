        def prepare_single_split(flag, df_split=None):
            encoder_path, shuffle_path = get_data_new_path(origin_datapath, flag=flag)

            if not os.path.exists(encoder_path) and df_split is not None:
                loader = DataLoader(DTIDataset(df_split), batch_size=self.batch_size, shuffle=True)
                encoder = SequenceEncoder(loader, self.feature_extractor)
                encoder.encode_and_save(encoder_path, shuffle_path)
            
            # 重点：返回一个组合了特征和序列的 Dataset
            combined_ds = PrecomputedCombinedDataset(encoder_path, shuffle_path)
            return DataLoader(combined_ds, batch_size=self.batch_size, shuffle=True, collate_fn = collate_fn) # 这里可以放心 Shuffle
        