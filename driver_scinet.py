import os
import torch
from datetime import datetime
from experiments.exp_pems import Exp_pems
import argparse
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Config():
    def __init__(self, config_dict):
        self.__dict__ = config_dict
    def __repr__(self, ):
        return str(self.__dict__)

config_dict = {
    'dataset': 'custom',
    'dataset_name': 'custom',
     'norm_method': 'z_score',
     'normtype': 0,
     'use_gpu': False,
     'use_multi_gpu': False,
     'gpu': 0,
     'device': 'cuda:0',
     'window_size': 12,
     'horizon': 5,
     'input_dim': 18,
     'concat_len': 0,
     'single_step_output_One': 0,
     'train_length': 8,
     'valid_length': 2,
     'test_length': 0,
     'train': True,
     'resume': False,
     'evaluate': False,
     'finetune': False,
     'validate_freq': 1,
     'epoch': 80,
     'lr': 0.001,
     'batch_size': 8,
     'optimizer': 'N',
     'early_stop': False,
     'exponential_decay_step': 5,
     'decay_rate': 0.5,
     'lradj': 1,
     'weight_decay': 1e-05,
     'model_name': 'SCINet',
     'hidden_size': 0.0625,
     'INN': 1,
     'kernel': 5,
     'dilation': 1,
     'positionalEcoding': True,
     'groups': 1,
     'levels': 2,
     'stacks': 1,
     'dropout': 0.5,
     'RIN': False}

args = Config(config_dict)


'''
    'dataset': 'custom',
    'dataset_name': 'custom',
    'input_dim': 18
'''



#%%

torch.manual_seed(4321)  # reproducible
torch.cuda.manual_seed_all(4321)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
torch.backends.cudnn.enabled = True

Exp=Exp_pems
exp=Exp(args)

train_loader, valid_loader, node_cnt, test_normalize_statistic, val_normalize_statistic = exp._get_data()

exp.train()

#%%



if args.evaluate:
    before_evaluation = datetime.now().timestamp()
    exp.test()
    after_evaluation = datetime.now().timestamp()
    print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
elif args.train or args.resume:
    before_train = datetime.now().timestamp()
    print("===================Normal-Start=========================")
    _, normalize_statistic = exp.train()
    after_train = datetime.now().timestamp()
    print(f'Training took {(after_train - before_train) / 60} minutes')
    print("===================Normal-End=========================")

#%%

model = exp.model
# model.load_state_dict('soemthing_goes_here!!')


dataloader = valid_loader
node_cnt = args.input_dim
window_size = args.window_size
horizon = args.horizon



forecast_set = []
Mid_set = []
target_set = []
input_set = []
model.eval()
with torch.no_grad():
    for i, (inputs, target) in enumerate(dataloader):
        if args.use_gpu:
            inputs = inputs.cuda()
            target = target.cuda()
        else:
            inputs = inputs.cpu()
            target = target.cpu()
            
        input_set.append(inputs.detach().cpu().numpy())
        step = 0
        forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
        Mid_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
        while step < horizon:
            if args.stacks == 1:
                forecast_result = model(inputs)
            elif args.stacks == 2:
                forecast_result, Mid_result = model(inputs)

            len_model_output = forecast_result.size()[1]
            if len_model_output == 0:
                raise Exception('Get blank inference result')
            inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                            :].clone()
            inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
            forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
            if args.stacks == 2:
                Mid_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    Mid_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()

            step += min(horizon - step, len_model_output)
        forecast_set.append(forecast_steps)
        target_set.append(target.detach().cpu().numpy())
        if args.stacks == 2:
            Mid_set.append(Mid_steps)

        result_save = np.concatenate(forecast_set, axis=0)
        target_save = np.concatenate(target_set, axis=0)

    # return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0), np.concatenate(input_set, axis=0)
    # return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0),np.concatenate(Mid_set, axis=0), np.concatenate(input_set, axis=0)

