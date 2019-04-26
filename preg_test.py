from __future__ import print_function

import sys
import os
import torch
import torch.nn as nn
import numpy as np


from utils import registration_method

import pyreg.fileio as py_fio
import pyreg.utils as py_utils
from modules.pregis_net import PregisNet
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision.utils as utils
import json
from utils.utils import *


def test_model(model, test_data_loader, network_config, my_model_folder):
    result_folder = os.path.join(my_model_folder, 'test')
    os.system('mkdir -p {}'.format(result_folder))
    image_io = py_fio.ImageIO()
    map_io = py_fio.MapIO()
    _id = py_utils.identity_map_multiN(np.array(network_config['model']['img_sz']), network_config['model']['target_spacing'])
    identity_map = torch.from_numpy(_id).cuda()
    with torch.no_grad():
        model.eval()
        for i, (moving_image, target_image) in enumerate(test_data_loader):
            moving_image = moving_image.cuda()
            target_image = target_image.cuda()
            
            moving_warped, moving_warped_recons, phi = model(moving_image, target_image, current_epoch=0)

            batch_size = network_config['validate']['batch_size']
            for j in range(batch_size):
                moving_file = os.path.join(result_folder,'test_{}.nii.gz'.format(i*batch_size+j))
                warped_file = os.path.join(result_folder, 'warped_{}.nii.gz'.format(i*batch_size+j))
                recons_file = os.path.join(result_folder, 'recons_{}.nii.gz'.format(i*batch_size+j))
                diff_file = os.path.join(result_folder, 'diff_{}.nii.gz'.format(i*batch_size+j))
                map_file = os.path.join(result_folder, 'disp_map_{}.nii.gz'.format(i*batch_size+j))
                image_io.write(moving_file, torch.squeeze(moving_image[j,...]), hdr=network_config['model']['target_hdrc'])
                image_io.write(warped_file, torch.squeeze(moving_warped[j,...]), hdr=network_config['model']['target_hdrc'])
                image_io.write(recons_file, torch.squeeze(moving_warped_recons[j,...]), hdr=network_config['model']['target_hdrc'])
                image_io.write(diff_file, torch.squeeze(moving_warped[j,...]-moving_warped_recons[j,...]), hdr=network_config['model']['target_hdrc'])
                disp_map = phi[i,...] - identity_map[i,...]
                map_io.write(filename=map_file, data=torch.squeeze(disp_map), hdr=network_config['model']['target_hdrc'])
             
            

def test_network():
    model_folder = "tmp_models/vae/model_20190424-175734_sm1.000_gm_1.000_gr1.000_reconsTV" 
    model_path = os.path.join(model_folder, 'best_eval.pth.tar')
    mermaid_config_file = os.path.join(model_folder, 'mermaid_config.json')
    network_config_file = os.path.join(model_folder, 'network_config.json')
    with open(mermaid_config_file) as f:
        mermaid_config = json.load(f)

    with open(network_config_file) as f:
        network_config = json.load(f)    
    
    model_config = network_config['model']
    train_config = network_config['train']
    validate_config = network_config['validate']
    _, _, test_data_loader = create_dataloader(model_config, train_config, validate_config)
    model_config['mermaid_config_file'] = mermaid_config_file
    model = create_model(model_config)

    network_state = torch.load(model_path)
    model.load_state_dict(network_state['model_state_dict'])
    print(network_state['epoch'])
    test_model(model, test_data_loader, network_config, model_folder)



if __name__ == '__main__':
    test_network()
