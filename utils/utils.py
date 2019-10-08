import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../mermaid'))
import pyreg.fileio as py_fio

from modules.pregis_net import PregisNet
from modules.mermaid_net import MermaidNet
from torch.utils.data import DataLoader

from data_loaders import ct_cbct as CBCTDataset


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def create_model(network_config, network_mode):
    if network_mode == 'mermaid':
        model = MermaidNet(network_config)
    elif network_mode == 'pregis':
        model = PregisNet(network_config)
    model.cuda()
    model.apply(weights_init)
    return model


def create_dataset(dataset_name, dataset_type, mode):
    if dataset_name == "ct_cbct":
        dataset = CBCTDataset.CBCTDataset
        return dataset(dataset_type, mode)
    else:
        raise ValueError("dataset not available")


def create_dataloader(network_config):
    model_config = network_config['model']

    train_dataset = network_config['train']['dataset']
    train_dataset_type = network_config['train']['dataset_type']

    validate_dataset = network_config['validate']['dataset']
    validate_dataset_type = network_config['validate']['dataset_type']

    my_train_dataset = create_dataset(train_dataset, train_dataset_type, 'train')
    my_validate_dataset = create_dataset(validate_dataset, validate_dataset_type, 'validate')

    train_data_loader = DataLoader(my_train_dataset,
                                   batch_size=network_config['train']['batch_size'],
                                   shuffle=True, num_workers=1,
                                   drop_last=True)
    validate_data_loader = DataLoader(my_validate_dataset,
                                      batch_size=network_config['train']['batch_size'],
                                      shuffle=True,
                                      num_workers=1,
                                      drop_last=True)

    model_config['img_sz'] = [network_config['train']['batch_size'], 1] + list(my_train_dataset.sz)
    model_config['dim'] = len(my_train_dataset.sz)
    model_config['target_hdrc'] = my_train_dataset.target_hdrc
    model_config['target_spacing'] = my_train_dataset.target_spacing
    return train_data_loader, validate_data_loader


def create_optimizer(config, model):
    method = config['optimizer']['name']
    scheduler_name = config['optimizer']['scheduler']['name']
    if method == "ADAM":
        lr = config['optimizer']['lr']
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("optimizer not supported")

    if scheduler_name == 'multistepLR':
        milestones = list(config['optimizer']['scheduler']['milestones'])
        gamma = config['optimizer']['scheduler']['gamma']
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        raise ValueError("scheduler not supported")
    return optimizer, scheduler


def create_loss(config):
    name = config['loss']['name']
    reduction = config['loss']['reduction']

    if name == 'MSE':
        criterion = nn.MSELoss(reduction=reduction).cuda()
    elif name == 'L1':
        criterion = nn.L1Loss(reduction=reduction).cuda()
    else:
        print('Loss specified unrecognized. MSE Loss is used')
        criterion = nn.MSELoss(reduction=reduction).cuda()
    return criterion


if __name__ == '__main__':
    # This is to test the results from evaluate_model
    print('test')
