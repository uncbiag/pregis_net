from __future__ import print_function

import sys
import os
import torch

import torch.backends.cudnn as cudnn
cudnn.benchmark = True 

import datetime
from tensorboardX import SummaryWriter
from utils.utils import *
from utils.visualize import make_image_summary

import json



def train_model(model, train_data_loader, validate_data_loader, optimizer, scheduler,
                network_config, my_name, my_model_folder, model_path=None):
    n_epochs = network_config['train']['n_of_epochs']
    current_epoch = 0
    batch_size = network_config['train']['batch_size']
    iters_per_epoch = len(train_data_loader.dataset)//batch_size
    val_iters_per_epoch = len(validate_data_loader.dataset)//batch_size
    summary_batch_period = min(network_config['train']['min_summary_period'], iters_per_epoch)
    validate_epoch_period = network_config['train']['validate_epoch_period']
   
    print('batch_size:', str(batch_size))
    print('iters_per_epoch:', str(iters_per_epoch))
    print('val_iters_per_epoch:', str(val_iters_per_epoch))
    print('summary_batch_period:', str(summary_batch_period))
    print("validate_epoch_period:", str(validate_epoch_period))

    writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 'logs', my_name))

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
        epoch_loss_dict = {
            'all_loss': 0.0,
            'mermaid_reg_loss': 0.0,
            'kld_loss': 0.0,
            'mermaid_sim_loss': 0.0,
            'recons_sim_loss': 0.0,
            'sim_loss': 0.0,
            'recons_loss_L1': 0.0,
            'recons_loss_TV': 0.0,
            'recons_loss': 0.0
        }
        
        scheduler.step(epoch=current_epoch+1)
        for i, (moving_image,target_image) in enumerate(train_data_loader, 0):
            model.train()
            optimizer.zero_grad()
            global_step = current_epoch * iters_per_epoch + (i+1) 

            moving_image = moving_image.cuda()
            target_image = target_image.cuda()
            moving_warped, moving_warped_recons, phi = model(moving_image, target_image, current_epoch=current_epoch+1)
            loss_dict = model.cal_pregis_loss(moving_image, target_image, cur_epoch=current_epoch+1, mode='train')
            all_loss = loss_dict['all_loss']
            all_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            for loss_key in epoch_loss_dict:
                if loss_key in loss_dict:
                    epoch_loss_dict[loss_key] += loss_dict[loss_key].item()
                else:
                    print("Warning: {} not exist in loss_dict. Adding zero".format(loss_key))
                    epoch_loss_dict[loss_key] += 0.0

            if (i+1) % summary_batch_period == 0: # print summary every k batches
                print('====>{:0d}, {:0d}, loss:{:.6f}, rec_loss:{:.6f}, m_sim_loss:{:.6f}, r_sim_loss:{:.6f}, sim_loss:{:.6f}, lr:{:.6f}'.format(
                    current_epoch+1,
                    global_step,
                    epoch_loss_dict['all_loss']/summary_batch_period,
                    epoch_loss_dict['recons_loss']/summary_batch_period,
                    epoch_loss_dict['mermaid_sim_loss']/summary_batch_period,
                    epoch_loss_dict['recons_sim_loss']/summary_batch_period,
                    epoch_loss_dict['sim_loss']/summary_batch_period,
                    optimizer.param_groups[0]['lr'])
                )

                for loss_key in epoch_loss_dict:
                    writer.add_scalar('training/training_{}'.format(loss_key), epoch_loss_dict[loss_key]/summary_batch_period, global_step=global_step)

                image_summary = make_image_summary(moving_image, target_image, moving_warped, moving_warped_recons, phi)
                for key, value in image_summary.items():
                    writer.add_image("training_" + key, value, global_step=global_step)
                epoch_loss_dict = {
                    'all_loss': 0.0,
                    'mermaid_reg_loss': 0.0,
                    'kld_loss': 0.0,
                    'mermaid_sim_loss': 0.0,
                    'recons_sim_loss': 0.0,
                    'sim_loss': 0.0,
                    'recons_loss_L1': 0.0,
                    'recons_loss_TV': 0.0,
                    'recons_loss': 0.0
                }



        if current_epoch % validate_epoch_period == 0: #validate every k epochs
            with torch.no_grad():
                model.eval()
                eval_loss_dict = {
                    'all_loss': 0.0,
                    'mermaid_reg_loss': 0.0,
                    'kld_loss': 0.0,
                    'mermaid_sim_loss': 0.0,
                    'recons_sim_loss': 0.0,
                    'sim_loss': 0.0,
                    'recons_loss_L1': 0.0,
                    'recons_loss_TV': 0.0,
                    'recons_loss': 0.0
                }
                for j, (moving_image, target_image) in enumerate(validate_data_loader, 0):
                    moving_image = moving_image.cuda()
                    target_image = target_image.cuda()
                    moving_warped, moving_warped_recons, phi = model(moving_image, target_image, current_epoch=current_epoch+1)
                    
                    loss_dict = model.cal_pregis_loss(moving_image, target_image, cur_epoch=current_epoch+1, mode='validate')
                    for loss_key in eval_loss_dict:
                        if loss_key in loss_dict:
                            eval_loss_dict[loss_key] += loss_dict[loss_key].item()
                        else:
                            print("Warning: {} not exist in loss_dict. Adding zero".format(loss_key))
                            eval_loss_dict[loss_key] += 0.0


                print('=EVAL>{:0d}, {:0d}, loss:{:.6f}, rec_loss:{:.6f}, sim_loss:{:.6f}'.format(
                    current_epoch+1,
                    global_step,
                    eval_loss_dict['all_loss']/val_iters_per_epoch,
                    eval_loss_dict['recons_loss']/val_iters_per_epoch,
                    eval_loss_dict['sim_loss']/val_iters_per_epoch)
                )
                for loss_key in eval_loss_dict:
                    writer.add_scalar('validation/validation_{}'.format(loss_key), eval_loss_dict[loss_key]/val_iters_per_epoch, global_step=global_step)


                if min_val_loss == 0.0 and global_step >= 50:
                    min_val_loss = eval_loss_dict['all_loss']
                if eval_loss_dict['all_loss'] <= min_val_loss:
                    min_val_loss = eval_loss_dict['all_loss']
                    save_file = os.path.join(my_model_folder, 'best_eval.pth.tar')
                    print("Writing current best eval model")
                    torch.save({'epoch': current_epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},save_file)
                    image_summary = make_image_summary(moving_image, target_image, moving_warped, moving_warped_recons, phi)
                    for key, value in image_summary.items():
                        writer.add_image("validation_" + key, value, global_step=global_step)
        current_epoch = current_epoch+1


def train_network():

    now = datetime.datetime.now()
    my_time = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    is_continue = False

    model_folder = None
    if is_continue:
        model_folder = os.path.join(os.path.dirname(__file__), "/main/tmp_models/my_model_20190324-011739")
        my_name = model_folder.split('/')[-1]
    if model_folder is not None:
        network_config_file = os.path.join(model_folder, 'network_config.json')
        mermaid_config_file = os.path.join(model_folder, 'mermaid_config.json')
    else:
        network_config_file = os.path.join(os.path.dirname(__file__), "settings/network_config.json")
        mermaid_config_file= os.path.join(os.path.dirname(__file__), 'settings/mermaid_config.json')
    with open(mermaid_config_file) as f:
        mermaid_config = json.load(f)
    with open(network_config_file) as f:
        network_config = json.load(f)
    sigma = mermaid_config['model']['registration_model']['similarity_measure']['sigma']
    gamma_recons = network_config['model']['pregis_net']['recons_net']['gamma_recons']
    gamma_mermaid = network_config['model']['pregis_net']['momentum_net']['gamma_mermaid']
    use_TV_loss = network_config['model']['pregis_net']['recons_net']['use_TV_loss']
    join_two_networks = network_config['model']['pregis_net']['join_two_networks']
    model_path = None
    #if is_continue:
        #model_path = os.path.join(model_folder, '299.pth.tar')
         #model_path = 'tmp_models/tmp_model_direct_momentum_' + my_name + '_409.pth.tar'


    model_config = network_config['model']
    train_config = network_config['train']
    validate_config = network_config['validate']
    
    #target_file = model_config['target_file']
    #image_io = py_fio.ImageIO()
    #target_image, target_hdrc, target_spacing, _ = image_io.read_to_nc_format(target_file, silent_mode=True)
    #model_config['img_sz'] = [train_config['batch_size'],1 ] + list(target_image.shape[2:])
    
    train_data_loader, validate_data_loader, _ = create_dataloader(model_config, train_config, validate_config)
    model['mermaid_config_file'] = mermaid_config_file
    model = create_model(model_config)
    optimizer, scheduler = create_optimizer(train_config, model)
   
    #criterion = create_loss(train_config)
    if not is_continue:
        my_name = "model_{}_sm{:.3f}_gm{:.3f}_gr{:.3f}_loss{}_{}".format(
            my_time,
            sigma,
            gamma_mermaid,
            gamma_recons,
            "TV" if use_TV_loss else "L1",
            "Join" if join_two_networks else "NotJ")
        model_folder = os.path.join(os.path.dirname(__file__), 'tmp_models', 'vae', my_name)
        os.system('mkdir -p ' + model_folder)
        os.system('cp ' + network_config_file + ' ' + model_folder)
        os.system('cp ' + mermaid_config_file + ' ' + model_folder)
        log_folder = os.path.join(os.path.dirname(__file__), 'logs', my_name)
        os.system('mkdir -p ' + log_folder)


    train_model(model, train_data_loader, validate_data_loader, optimizer, scheduler, network_config, my_name, model_folder, model_path=model_path)

    return

if __name__ == '__main__':
    train_network()
