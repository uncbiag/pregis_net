from modules.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class AedNet(nn.Module):
    def __init__(self, dim=3):
        super(AedNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_bn_rel_dp(1, 8, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', use_bn=False),
            conv_bn_rel_dp(8, 16, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', use_bn=False),
            mp(kernel_size=2, dim=dim),
            conv_bn_rel_dp(16, 32, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu'),
            conv_bn_rel_dp(32, 32, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu'),
            mp(kernel_size=2, dim=dim),
            conv_bn_rel_dp(32, 64, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu'),
            conv_bn_rel_dp(64, 64, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu'),
            mp(kernel_size=2, dim=dim),
            conv_bn_rel_dp(64, 128, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu'),
        )
        self.decoder = nn.Sequential(
            conv_bn_rel_dp(128, 64, kernel_size=2, stride=2, dim=dim, reverse=True, activate_unit='leaky_relu'),
            conv_bn_rel_dp(64, 64, kernel_size=3, stride=1, dim=dim, reverse=False, activate_unit='leaky_relu'),
            conv_bn_rel_dp(64, 32, kernel_size=2, stride=2, dim=dim, reverse=True, activate_unit='leaky_relu'),
            conv_bn_rel_dp(32, 32, kernel_size=3, stride=1, dim=dim, reverse=False, activate_unit='leaky_relu'),
            conv_bn_rel_dp(32, 16, kernel_size=2, stride=2, dim=dim, reverse=True, activate_unit='leaky_relu'),
            conv_bn_rel_dp(16, 8, kernel_size=3, stride=1, dim=dim, reverse=False, activate_unit='None'),
            conv_bn_rel_dp(8, 1, kernel_size=3, stride=1, dim=dim, reverse=False, activate_unit='None'),
            nn.Sigmoid()
        )
        return


    def forward(self, input):
        h = self.encoder(input)
        output = self.decoder(h)
 
        return output


