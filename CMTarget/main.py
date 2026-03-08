import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import argparse
import json
from datetime import datetime
from datasets.preprocess import *
from trainer.CMTargetTrainer import CMTargetTrainer
from predictor.CMTargetPredictor import CMTargetPredictor
from utils.utils import *
from datasets.preextract import *

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', default='drugbank')
    parser.add_argument('-t', '--target', default='hit')
    parser.add_argument('-m', '--model', default='AttFusion')
    parser.add_argument('-emb', '--embedding_dim', type=int, default=512)
    parser.add_argument('--meta_dim', type=int, default=512)
    parser.add_argument('-hl', '--history_length', type=int, default=20)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-mp', '--model_path', default='')
    parser.add_argument('--task', default='train')
    parser.add_argument('-d', '--data_dir', default='cdr')
    parser.add_argument('-pe', '--protein_encoder', default='dict')
    parser.add_argument('-de', '--drug_encoder', default='fingerprint')
    parser.add_argument('--score', default='MF')
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    args = parser.parse_args()

    config = json.load(f)
    config['source'] = args.source
    config['target'] = args.target
    config['model'] = args.model
    config['his_len'] = args.history_length
    config['epoch'] = args.epoch
    config['lr'] = args.learning_rate
    config['wd'] = args.weight_decay
    config['gpu'] = args.gpu
    config['emb'] = args.embedding_dim
    config['meta_dim'] = args.meta_dim
    config['batch_size'] = args.batch_size
    config['timestamp'] = datetime.now().strftime("%Y%m%d%H%M%S")
    config['model_path'] = args.model_path
    config['task'] = args.task
    config['data_dir'] = args.data_dir
    config['num_workers'] = args.num_workers
    config['score'] = args.score

    return config

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    configs = prepare()
    log_path = 'logs/{}'.format(configs['timestamp'])

    config_dir = 'configs'
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    config_path = os.path.join(config_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(configs, f, indent=4)

    if configs['task'] == 'train':
        print("start training")
        print(f"train model {configs['model']}: epoch: {configs['epoch']}, batch_size: {configs['batch_size']}, lr: {configs['lr']}")

        if configs['model'] == 'CMTarget':
            trainer = CMTargetTrainer(configs)
            trainer.train()

    elif configs['task'] == 'predict':
        print("start Predicting")
        if configs['model'] == 'CMTarget':
            predictor = CMTargetPredictor(configs)
            predictor.predict()

    elif configs['task'] == 'split':
        print("start split {} and {}".format(configs['source'], configs['target']))
        data_processor = CDRDataPreprocessor(configs['source'], configs['target'], configs['data_dir'], configs['his_len'])
        data_processor.run()
        print("split done!")
    elif configs['task'] == 'extract':
        print("start extract {} and {}".format(configs['source'], configs['target']))
        prot_bert_extract(configs)
        drug_bert_extract(configs)
        print("extract done!")
