from __future__ import print_function

import sys
import os
import torch
import torch.nn as nn
import numpy as np


from utils import registration_method
#mermaid_path='/playpen/xhs400/Research/FPIR/mermaid'
#sys.path.append(mermaid_path)
#sys.path.append(os.path.join(mermaid_path, 'pyreg'))
#sys.path.append(os.path.join(mermaid_path, 'pyreg/libraries'))
#sys.path.append('./modules')
import pyreg.fileio as py_fio
from modules.pregis_net import PregisNet
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision.utils as utils
import json
from utils.utils import *


def test_model(model, validate_data_loader, config, my_model_folder):
    result_folder = os.path.join(my_model_folder, 'test')
    os.system('mkdir -p {}'.format(result_folder))
    image_io = py_fio.ImageIO()
    with torch.no_grad():
        model.eval()
        for i, (moving_image, target_image) in enumerate(validate_data_loader):
            moving_image = moving_image.cuda()
            target_image = target_image.cuda()

            moving_warped, moving_warped_recons, phi = model(moving_image, target_image)

            batch_size = config['train']['batch_size']
            for j in range(batch_size):
                moving_file = os.path.join(result_folder,'test_{}.nii.gz'.format(i*batch_size+j))
                warped_file = os.path.join(result_folder, 'warped_{}.nii.gz'.format(i*batch_size+j))
                recons_file = os.path.join(result_folder, 'recons_{}.nii.gz'.format(i*batch_size+j))
                image_io.write(moving_file, torch.squeeze(moving_image[j,...]), hdr=config['model']['target_hdrc'])
                image_io.write(warped_file, torch.squeeze(moving_warped[j,...]), hdr=config['model']['target_hdrc'])
                image_io.write(recons_file, torch.squeeze(moving_warped_recons[j,...]), hdr=config['model']['target_hdrc'])
             
            

def test_network():
    model_folder = "tmp_models/vae/my_model_20190416-223818_sigma0.500_gr1.000_reconsL1" 
    model_path = os.path.join(model_folder, 'best_eval.pth.tar')
    mermaid_config_file = os.path.join(model_folder, 'mermaid_config.json')
    config_file = os.path.join(model_folder, 'network_config.json')
    with open(mermaid_config_file) as f:
        mermaid_config = json.load(f)

    with open(config_file) as f:
        config = json.load(f)    
    
    model_config = config['model']
    train_config = config['train']
    validate_config = config['validate']
    _, validate_data_loader = create_dataloader(model_config, train_config, validate_config)
 
    #image_io = py_fio.ImageIO()
    #atlas_file = validate_data_loader.atlas_file

    #target_image, target_hdrc, target_spacing,_ = image_io.read_to_nc_format(atlas_file)
    model = create_model(model_config)

    network_state = torch.load(model_path)
    model.load_state_dict(network_state['model_state_dict'])

    test_model(model, validate_data_loader, config, model_folder)



if __name__ == '__main__':
    test_network()
