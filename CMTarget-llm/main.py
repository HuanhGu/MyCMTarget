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
from predictor.CMTargetPredictor import CMTargetPredictor

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
    parser.add_argument('--source_path', type = str, default="")#模型路径

    parser.add_argument('-s', '--source', default='drugbank')
    parser.add_argument('-t', '--target', default='hit')

    parser.add_argument('--timestamp', type=str, default = "001")
    parser.add_argument('--task', type=str, default = "predict")#train

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
    config['source'] = args.source
    config['target'] = args.target
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

    # 2. 预处理序列数据，得到encoder
    # 序列文件路径(原始文件)
    drugbank_path = Path("./data/dataset/drugbank/drugbank.csv")
    hit_path = Path("./data/dataset/hit/hit.csv")
    
    # encoder后保存路径
    hit_encoder_path = Path("./data/encoder/hit_feature.pt")
    drugbank_encoder_path = Path("./data/encoder/drugbank_feature.pt")

    hit_shuffle_path = Path("./data/data_shuffle/hit_shuffle.csv")
    drugbank_shuffle_path = Path("./data/data_shuffle/drugbank_shuffle.csv")


    # data预处理 & 保存
     # 尽可能使用大的batch_size提高速度
    if not os.path.exists(hit_encoder_path):
        print(f"❌{hit_encoder_path}不存在encoder后的文件~")
        data_preEncoder(hit_path, hit_encoder_path, hit_shuffle_path,
                        drugbank_path, drugbank_encoder_path, drugbank_shuffle_path,
                        bs=32)


    # 3. 训练模型
    if configs['task'] == 'train':
        print("start training")
        print(f"train model {configs['model']}: epoch: {configs['epochs']}, batch_size: {configs['batch_size']}, lr: {configs['learning_rate']}")

        trainer = CMTargetTrainer(configs, hit_encoder_path, hit_shuffle_path)# model, 
        trainer.train()

        print("\nTraining finished.")
    
    elif configs['task'] == 'predict':
        print("start Predicting")

        predictor = CMTargetPredictor(configs, drugbank_encoder_path, hit_shuffle_path)#model
        predictor.predict()


        # 不建议直接把model当参数传入, 而是使用get_model；因为model也可以从本地保存的文件加载
        # 也就是train和pred任务分离

