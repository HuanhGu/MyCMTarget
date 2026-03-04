import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader

from embedding.dataset_build import *
from model.scorer import *
from model.CMTargetModel import *
from trainer.CMTargetTrainer import *





def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', default='drugbank')
    parser.add_argument('-t', '--target', default='hit')
    parser.add_argument('-emb', '--embedding_dim', type=int, default=512)
    parser.add_argument('-pTok', '--protein_encoder_Token_num', type=int, default=416)

    parser.add_argument('-lr', '--learning_rate', type=float, default = 0.0001)
    parser.add_argument('-ep', '--epochs', type=int, default = 2)
    parser.add_argument('-bs', '--batch_size', type = int, default = 3)

    parser.add_argument('-scW', '--score_way', type=str, default='MF') #打分器选择
    # parser.add_argument('-scD', '--score_emb_dim', type = int, default = 256)

    parser.add_argument('--task', type=str, default = "predict")#train
    parser.add_argument('-mod', '--model_name', type=str, default = "CMTarget")
    parser.add_argument('--model_path', type = str, default="")
    parser.add_argument('--source_path', type = str, default="")

    parser.add_argument('--timestamp', type=str, default = "001")

    args = parser.parse_args()

    
    config = {}
    config['source'] = args.source
    config['target'] = args.target
    config['emb'] = args.embedding_dim
    config['learning_rate'] = args.learning_rate
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size

    config['score_way'] = args.score_way
    # config['score_dim'] = args.score_emb_dim

    config['task'] = args.task
    config['model'] = args.model_name
    config['model_path'] = args.model_path

    config['timestamp'] = args.timestamp
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    return config

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    
    # 2. 获取超参数配置 ./configs/config.json
    configs = prepare()
    

    # config_dir = 'configs'
    # if not os.path.exists(config_dir):
    #     os.makedirs(config_dir)
    # config_path = os.path.join(config_dir, 'config.json')
    # with open(config_path, 'w') as f:
    #     json.dump(configs, f, indent=4)
    

    # 2. 读取序列数据,创建dataloader
    # 序列文件路径(原始文件)
    drugbank_path = Path("./data/dataset/drugbank/drugbank.csv")
    hit_path = Path("./data/dataset/hit/hit.csv")

    # 序列dataloader
    hit_dataset = DTIDataset(hit_path)
    hit_dataloader = DataLoader(hit_dataset, batch_size=2, shuffle=True, num_workers=0) # 序列的loader
    
    drugbank_dataset = DTIDataset(drugbank_path)
    drugbank_dataloader = DataLoader(drugbank_dataset, batch_size=2, shuffle=True, num_workers=0)
    
    

    # 3. 训练模型
    if configs['task'] == 'train':
        print("start training")
        print(f"train model {configs['model']}: epoch: {configs['epochs']}, batch_size: {configs['batch_size']}, lr: {configs['learning_rate']}")
        #原来直接这样就能格式化输出
        

        trainer = CMTargetTrainer(configs, hit_dataloader)# model, 
        trainer.train()

        print("\nTraining finished.")

    elif configs['task'] == 'predict':
        print("start Predicting")

        # 不建议直接把model当参数传入, 而是使用get_model；因为model也可以从本地保存的文件加载
        # 也就是train和pred任务分离

        model = CMTargetModel(configs)
        predictor = CMTargetTrainer(configs, drugbank_dataloader)#model, 
        predictor.model_evaluate(model, drugbank_dataloader)

