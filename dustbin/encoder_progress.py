
'''
# 2. 训练加载
train_ds = EncodedDTIDataset("features.pt")
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

for d, p, l in train_loader:
    # d 的形状: [32, max_drug_len_in_batch, 768]
    # p 的形状: [32, max_pro_len_in_batch, 100]
    pass

'''




''' 3. 构建 dataloader 

# 文件路径
drugbank_path = "./data/dataset/drugbank/drugbank.csv"
# drugbank_data = pd.read_csv(drugbank_path, sep="\t")

hit_path = "./data/dataset/hit/hit.csv"
# hit_data = pd.read_csv(hit_path, sep="\t")




hit_dataset = DTIDataset(hit_path)

hit_dataloader = DataLoader(
    hit_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)


# for compound_batch, protein_batch, label_batch in hit_dataloader:
#     print(len(compound_batch))   # 32
#     print(len(protein_batch))    # 32
#     print(label_batch.shape)     # torch.Size([32])

'''