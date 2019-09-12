import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../mermaid'))
import pyreg.fileio as py_fio

from modules.pregis_net import PregisNet
from torch.utils.data import DataLoader

from data_loaders import pseudo_2D as pseudo_2D_dataset
from data_loaders import pseudo_3D as pseudo_3D_dataset
from data_loaders import brats_3D as brats_3D_dataset


def collate_fn_cbctdataset(batch):
    moving_image, moving_label, target_image, target_label, momentum, target_spacing = zip(*batch)
    m_image = torch.from_numpy(moving_image[0])
    m_label = torch.from_numpy(moving_label[0])
    t_image = torch.from_numpy(target_image[0])
    t_label = torch.from_numpy(target_label[0])
    momentum = torch.from_numpy(momentum[0])
    return m_image, m_label, t_image, t_label, momentum, target_spacing[0]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def create_model(network_config, network_mode):
    model = PregisNet(network_config, network_mode)
    model.cuda()
    model.apply(weights_init)
    return model


def create_dataloader(model_config, tr_config, network_mode):
    dataset = model_config['dataset']
    print("Dataset to load: {}".format(dataset))
    if dataset == "brats_3D":
        MyDataset = brats_3D_dataset.Brats3DDataset
    elif dataset == "pseudo_2D":
        MyDataset = pseudo_2D_dataset.Pseudo2DDataset
    elif dataset == "pseudo_3D":
        MyDataset = pseudo_3D_dataset.Pseudo3DDataset
    else:
        raise ValueError("dataset not available")

    my_train_dataset = MyDataset('training', network_mode)
    my_validate_dataset = MyDataset('validation', network_mode)
    atlas_file = my_train_dataset.atlas_file
    print(atlas_file)
    image_io = py_fio.ImageIO()
    target_image, target_hdrc, target_spacing, _ = image_io.read_to_nc_format(atlas_file, silent_mode=True)

    train_data_loader = DataLoader(my_train_dataset,
                                   batch_size=tr_config['batch_size'],
                                   shuffle=True, num_workers=4,
                                   drop_last=True)
    validate_data_loader = DataLoader(my_validate_dataset,
                                      batch_size=tr_config['batch_size'],
                                      shuffle=False,
                                      num_workers=4,
                                      drop_last=True)
    # test_data_loader = DataLoader(my_test_dataset,
    #                              batch_size=va_config['batch_size'],
    #                              shuffle=False,
    #                              num_workers=4,
    #                              drop_last=True)
    model_config['img_sz'] = [tr_config['batch_size'], 1] + list(target_image.shape[2:])
    model_config['dim'] = len(target_image.shape[2:])
    model_config['target_hdrc'] = target_hdrc
    model_config['target_spacing'] = target_spacing
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
