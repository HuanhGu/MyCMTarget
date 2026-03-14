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
from trainer.CMTargetTrainer import CMTargetTrainer
from predictor.CMTargetPredictor import CMTargetPredictor
from fineTuner.FineTunner import FineTunner
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")




def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type = int, default = 3)
    parser.add_argument('-emb', '--embedding_dim', type=int, default=512)
    parser.add_argument('-ep', '--epochs', type=int, default = 2)

    parser.add_argument('-lr', '--learning_rate', type=float, default = 0.0001)
    parser.add_argument('-mod', '--model_name', type=str, default = "CMTarget")
    parser.add_argument('--model_path', type = str, default="")

    parser.add_argument('-pTok', '--protein_encoder_Token_num', type=int, default=416)

    parser.add_argument('-scW', '--score_way', type=str, default='MF') #打分器选择
    # parser.add_argument('-scD', '--score_emb_dim', type = int, default = 256)

    parser.add_argument('--source_datapath', type = str, default="./data/dataset/drugbank/drugbank.csv")
    parser.add_argument('--target_datapath', default='./data/dataset/hit/hit.csv')

    parser.add_argument('--timestamp', type=str, default = "001")
    parser.add_argument('--task', type=str, default = "train")#train\predict\tune

    args = parser.parse_args()

    config = {}
    config['batch_size'] = args.batch_size
    config['emb'] = args.embedding_dim
    config['epochs'] = args.epochs  
    config['learning_rate'] = args.learning_rate

    config['model'] = args.model_name
    config['model_path'] = args.model_path

    config['score_way'] = args.score_way
    # config['score_dim'] = args.score_emb_dim
    config['source_datapath'] = args.source_datapath
    config['target_datapath'] = args.target_datapath
    config['task'] = args.task

    config['timestamp'] = timestamp
    
    return config

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    
    # 2. 获取超参数配置 ./configs/config.json
    configs = prepare()
    
    config_dir = os.path.join('configs', configs['timestamp'])
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    config_path = os.path.join(config_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=4)
    
    configs['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 3. 训练模型
    
    if configs['task'] == 'train':
        os.makedirs(f"checkpoints/{configs['timestamp']}", exist_ok=True)
        pretrain_output_path = f"checkpoints/{configs['timestamp']}/pretrain.pt"
        fintune_output_path = f"checkpoints/{configs['timestamp']}/fineTune.pt"

        print(f"train model {configs['model']}: epoch: {configs['epochs']}, batch_size: {configs['batch_size']}, lr: {configs['learning_rate']}")
        
        trainer = CMTargetTrainer(configs, configs['source_datapath'], configs['model_path'])
        trainer.train(pretrain_output_path)
        
        # 加载pre_train完毕后的model_path, 作为初始值
        fineTunner = FineTunner(configs, configs['target_datapath'], configs['model_path'])#model
        fineTunner.fineTune(fintune_output_path)
        

    elif configs['task'] == 'predict':
        print("please check the fintune_output_path, and you must choose the correct path and varify it. ")
        
        predictor = CMTargetPredictor(configs, configs['model_path'])#model
        predictor.predict()
        
# 111111111111

