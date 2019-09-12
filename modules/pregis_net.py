import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append('../')
from modules.mermaid_net import MermaidNet
from modules.recons_net import ReconsNet


class PregisNet(nn.Module):
    def __init__(self, model_config, network_mode):
        super(PregisNet, self).__init__()
        self.mermaid_net = None
        self.recons_net = None
        self.network_mode = network_mode

        if self.network_mode in ['mermaid', 'pregis']:
            self.mermaid_net = MermaidNet(model_config, network_mode)
        if self.network_mode in ['recons', 'pregis']:
            self.recons_net = ReconsNet(model_config, network_mode)

        # results
        self.warped_image = None
        self.phi = None
        self.recons_image = None

        return

    #########################
    #  Calculate Loss functions
    #########################
    def cal_pregis_loss(self, moving, target, current_epoch):
        if self.network_mode == 'pregis':
            mermaid_loss = self.mermaid_net.cal_mermaid_loss(moving, target)
            vae_loss = self.recons_net.calculate_vae_loss(self.warped_image, target, current_epoch)
            all_loss = mermaid_loss['mermaid_all_loss'] + vae_loss['vae_all_loss']
            loss_dict = {**mermaid_loss, **vae_loss}
            loss_dict['all_loss'] = all_loss
        elif self.network_mode == 'mermaid':
            loss_dict = self.mermaid_net.cal_mermaid_loss(moving, target)
        elif self.network_mode == 'recons':
            loss_dict = self.recons_net.calculate_vae_loss(moving, target, current_epoch)
        else:
            raise ValueError("Network mode not correct")

        return loss_dict

    #####################################
    #  Forward pass
    #####################################

    def single_forward(self, moving, target):

        if self.network_mode == 'pregis':
            self.warped_image, self.phi = self.mermaid_net(moving, target)
            self.recons_image = self.recons_net(self.warped_image, target)
        elif self.network_mode == 'mermaid':
            self.warped_image, self.phi = self.mermaid_net(moving, target)
        elif self.network_mode == 'recons':
            self.recons_image = self.recons_net(moving, target)
        else:
            raise ValueError("Network option is not correct.")

        return self.warped_image, self.recons_image, self.phi

    def forward(self, moving, target):
        self.single_forward(moving, target)
        return
