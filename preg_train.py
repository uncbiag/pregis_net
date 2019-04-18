from __future__ import print_function

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
import numpy as np

import torch.backends.cudnn as cudnn
cudnn.benchmark = True 

from utils import registration_method
mermaid_path='/playpen/xhs400/Research/FPIR/mermaid'
sys.path.append(mermaid_path)
sys.path.append(os.path.join(mermaid_path, 'pyreg'))
sys.path.append(os.path.join(mermaid_path, 'pyreg/libraries'))
import pyreg.fileio as py_fio
import datetime
import time


from data_loaders import pseudo_2D as pseudo_2D_dataset
from data_loaders import brats_3D as brats_3D_dataset
from data_loaders import pseudo_3D as pseudo_3D_dataset
#sys.path.append('./modules')
from modules.pregis_net import PregisNet

from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision.utils as utils
#from losses import loss
from utils.utils import *

from utils.visualize import make_image_summary
import json




def train_model(model, train_data_loader, validate_data_loader, optimizer, scheduler,
                config, my_name, my_model_folder, model_path=None):
    n_epochs = 10000
    current_epoch = 0
    batch_size = config['train']['batch_size']
    iters_per_epoch = len(train_data_loader.dataset)//batch_size
    val_iters_per_epoch = len(validate_data_loader.dataset)//batch_size
    summary_batch_num = min(5, iters_per_epoch)
    print('batch_size:', str(batch_size))
    print('iters_per_epoch:', str(iters_per_epoch))
    print('val_iters_per_epoch:', str(val_iters_per_epoch))
    print('summery_batch_num:', str(summary_batch_num))


    writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 'new-logs', my_name))

    if model_path is not None:
        # resume training
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            if 'epoch' in checkpoint:
                current_epoch = checkpoint['epoch'] + 1
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                print("Model load FAILED")
            if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    min_val_loss = 0.0
    while current_epoch < n_epochs:
        epoch_loss = 0.0
        epoch_sim_loss = 0.0
        epoch_rec_loss = 0.0
        epoch_rec_sim_loss = 0.0
        epoch_kld_loss = 0.0
        epoch_mermaid_sim_loss = 0.0
        epoch_mermaid_reg_loss = 0.0
        scheduler.step(epoch=current_epoch+1)
        m_sum = 0.0
        for i, (moving_image,target_image) in enumerate(train_data_loader, 0):
            model.train()
            optimizer.zero_grad()
            global_step = current_epoch * iters_per_epoch + (i+1) 

            moving_image = moving_image.cuda()
            target_image = target_image.cuda()
            moving_warped, moving_warped_recons, phi = model(moving_image, target_image)
            all_loss, mermaid_reg_loss, sim_loss, kld_loss, recons_loss, recons_sim_loss, mermaid_sim_loss = model.cal_pregis_loss(moving_image, target_image, cur_epoch=current_epoch+1)
            all_loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + all_loss.item()
            epoch_rec_loss += recons_loss.item()
            epoch_kld_loss += kld_loss.item()
            epoch_rec_sim_loss += recons_sim_loss.item()
            epoch_sim_loss += sim_loss.item()
            epoch_mermaid_sim_loss += mermaid_sim_loss.item()
            epoch_mermaid_reg_loss += mermaid_reg_loss.item()
            
            if (i+1) % summary_batch_num == 0: # print summary every k batches
                print('====>{:0d}, {:0d}, loss:{:.6f}, rec_loss:{:.6f}, m_sim_loss:{:.6f}, r_sim_loss:{:.6f}, sim_loss:{:.6f}, lr:{:.6f}'.format(current_epoch+1, global_step, epoch_loss/summary_batch_num, epoch_rec_loss/summary_batch_num, epoch_mermaid_sim_loss/summary_batch_num, epoch_rec_sim_loss/summary_batch_num, epoch_sim_loss/summary_batch_num,optimizer.param_groups[0]['lr']))
                writer.add_scalar('training/training_loss', epoch_loss/summary_batch_num, global_step=global_step)
                writer.add_scalar('training/training_mermaid_reg_loss', epoch_mermaid_reg_loss/summary_batch_num, global_step=global_step)
                writer.add_scalar('training/training_sim_loss', epoch_sim_loss/summary_batch_num, global_step=global_step)
                writer.add_scalar('training/training_kld_loss', epoch_kld_loss/summary_batch_num, global_step=global_step)
                writer.add_scalar('training/training_recons_loss', epoch_rec_loss/summary_batch_num, global_step=global_step)
                writer.add_scalar('training/training_recons_sim_loss', epoch_rec_sim_loss/summary_batch_num, global_step=global_step)
                writer.add_scalar('training/training_mermaid_sim_loss', epoch_mermaid_sim_loss/summary_batch_num, global_step=global_step)

                image_summary = make_image_summary(moving_image, target_image, moving_warped, moving_warped_recons, phi)
                for key, value in image_summary.items():
                    writer.add_image("training_" + key, value, global_step=global_step)
                epoch_loss = 0.0
                epoch_sim_loss = 0.0
                epoch_rec_sim_loss = 0.0
                epoch_rec_loss = 0.0
                epoch_kld_loss = 0.0
                epoch_mermaid_sim_loss = 0.0
                epoch_mermaid_reg_loss = 0.0
    
        if current_epoch % 10 == 0: #validate every k epochs
            with torch.no_grad():
                model.eval()
                eval_loss = 0.0
                eval_rec_loss = 0.0
                eval_mermaid_reg_loss = 0.0
                eval_sim_loss = 0.0
                eval_kld_loss = 0.0
                for j, (moving_image, target_image) in enumerate(validate_data_loader, 0):
                    moving_image = moving_image.cuda()
                    target_image = target_image.cuda()
                    moving_warped, moving_warped_recons, phi = model(moving_image, target_image, fix_momentum=False)
                    all_loss, mermaid_reg_loss, sim_loss, kld_loss, recons_loss, _, _ = model.cal_pregis_loss(moving_image, target_image, cur_epoch=current_epoch+1)
                    eval_loss = eval_loss + all_loss.item()
                    eval_rec_loss += recons_loss.item()
                    eval_mermaid_reg_loss += mermaid_reg_loss.item()
                    eval_kld_loss += kld_loss.item()
                    eval_sim_loss += sim_loss.item()

                print('=EVAL>{:0d}, {:0d}, loss:{:.6f}, rec_loss:{:.6f}, sim_loss'.format(current_epoch+1, global_step, eval_loss/val_iters_per_epoch, eval_rec_loss/val_iters_per_epoch, eval_sim_loss/val_iters_per_epoch))
                writer.add_scalar('validation/validation_loss', eval_loss/val_iters_per_epoch, global_step=global_step)
                writer.add_scalar('validation/validation_recons_loss', eval_rec_loss/val_iters_per_epoch, global_step=global_step)
                writer.add_scalar('validation/validation_kld_loss', eval_kld_loss/val_iters_per_epoch, global_step=global_step)
                writer.add_scalar('validation/validation_mermaid_reg_loss', eval_mermaid_reg_loss/val_iters_per_epoch, global_step=global_step)
                writer.add_scalar('validation/validation_sim_loss', eval_sim_loss/val_iters_per_epoch, global_step=global_step)
                

                if min_val_loss == 0.0 and global_step >= 50:
                    min_val_loss = eval_loss
                if eval_loss <= min_val_loss:
                    min_val_loss = eval_loss
                    save_file = os.path.join(my_model_folder, 'best_eval.pth.tar')
                    print("Writing current best eval model")
                    torch.save({'epoch': current_epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},save_file)
                    image_summary = make_image_summary(moving_image, target_image, moving_warped, moving_warped_recons, phi)
                    for key, value in image_summary.items():
                        writer.add_image("validation_" + key, value, global_step=global_step)
        current_epoch = current_epoch+1


def create_model(config):
    name = config['name']
    if name == 'pregis_net':
        #config = config[name]
        n_of_feature = config['n_of_feature']
        bn = config['bn']
        dp = config['dp']
        img_sz = config['img_sz']
        gamma_recons = config[name]['gamma_recons']
        dim = config['dim']
        recons_loss = config[name]['recons_loss']
        model = PregisNet(config)
    else:
        raise ValueError("Model Not supported")
    model.cuda()
    model.apply(weights_init)
    return model


def create_dataloader(model_config, tr_config, va_config):
    dataset = model_config['dataset']
    if dataset == "brats_3D":
        MyDataset = brats_3D_dataset.Brats3DDataset
    elif dataset == "pseudo_2D":
        MyDataset = pseudo_2D_dataset.Pseudo2DDataset
    elif dataset == "pseudo_3D":
        MyDataset = pseudo_3D_dataset.Pseudo3DDataset
    else:
        raise ValueError("dataset not available")
    
    my_train_dataset = MyDataset(mode='training')
    my_validate_dataset = MyDataset(mode='validation')
    atlas_file = my_train_dataset.atlas_file
    print(atlas_file)
    image_io = py_fio.ImageIO()
    target_image, _, _, _ = image_io.read_to_nc_format(atlas_file, silent_mode=True)

    train_data_loader = DataLoader(my_train_dataset, batch_size=tr_config['batch_size'], shuffle=True)
    validate_data_loader = DataLoader(my_validate_dataset, batch_size=va_config['batch_size'], shuffle=False)
    model_config['img_sz'] = [tr_config['batch_size'], 1] + list(target_image.shape[2:])
    model_config['dim'] = len(target_image.shape[2:])
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
    #elif name == 'L12':
        #criterion = loss.L12Loss().cuda()
    else:
        print('Loss specified unrecognized. MSE Loss is used')
        criterion = nn.MSELoss(reduction=reduction).cuda()
    return criterion



def train_network():

    now = datetime.datetime.now()
    my_time = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    is_continue = False

    model_folder = None
    if is_continue:
        model_folder = "/playpen/xhs400/Research/PycharmProjects/pregis_net/main/tmp_models/my_model_20190324-011739"
        my_name = model_folder.split('/')[-1]
    if model_folder is not None:
        config_file = os.path.join(model_folder, 'network_config.json')
    else:
        config_file = "/playpen/xhs400/Research/PycharmProjects/pregis_net/main/settings/network_config.json"
    mermaid_config_file= os.path.join(os.path.dirname(__file__), 'settings/mermaid_config.json')
    with open(mermaid_config_file) as f:
        mermaid_config = json.load(f)
    with open(config_file) as f:
        config = json.load(f)
    sigma = mermaid_config['model']['registration_model']['similarity_measure']['sigma']
    gamma_recons = config['model']['pregis_net']['gamma_recons']
    recons_loss = config['model']['pregis_net']['recons_loss']

    model_path = None
    #if is_continue:
        #model_path = os.path.join(model_folder, '299.pth.tar')
         #model_path = 'tmp_models/tmp_model_direct_momentum_' + my_name + '_409.pth.tar'


    model_config = config['model']
    train_config = config['train']
    validate_config = config['validate']
    
    #target_file = model_config['target_file']
    #image_io = py_fio.ImageIO()
    #target_image, target_hdrc, target_spacing, _ = image_io.read_to_nc_format(target_file, silent_mode=True)
    #model_config['img_sz'] = [train_config['batch_size'],1 ] + list(target_image.shape[2:])
    
    train_data_loader, validate_data_loader = create_dataloader(model_config, train_config, validate_config)
    model = create_model(model_config)
    optimizer, scheduler = create_optimizer(train_config, model)
   
    #criterion = create_loss(train_config)
    if not is_continue:
        my_name = "my_model_{}_sigma{:.3f}_gr{:.3f}_recons{}".format(
            my_time,
            sigma,
            gamma_recons,
            recons_loss)
        model_folder = os.path.join(os.path.dirname(__file__), 'tmp_models', 'vae', my_name)
        os.system('mkdir -p ' + model_folder)
        os.system('cp ' + config_file + ' ' + model_folder)
        os.system('cp ' + mermaid_config_file + ' ' + model_folder)
        log_folder = os.path.join(os.path.dirname(__file__), 'new-logs', my_name)
        os.system('mkdir -p ' + log_folder)


    train_model(model, train_data_loader, validate_data_loader, optimizer, scheduler, config, my_name, model_folder, model_path=model_path)

    return

if __name__ == '__main__':
    train_network()
