from modules.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class VaeNet(nn.Module):
    def __init__(self, dim=3):
        super(VaeNet, self).__init__()
        self.dim= dim

        #convolution to get mu and logvar
        self.conv11 = conv_bn_rel_dp(64, 128, kernel_size=3, stride=2, dim=dim, activate_unit='None', same_padding=True)
        self.conv12 = conv_bn_rel_dp(64, 128, kernel_size=3, stride=2, dim=dim, activate_unit='None', same_padding=True)

        self.encoder = nn.Sequential(
            conv_bn_rel_dp(1, 8, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', same_padding=True, use_bn=False),
            conv_bn_rel_dp(8, 16, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', same_padding=True, use_bn=False),
            mp(kernel_size=2, dim=dim),
            conv_bn_rel_dp(16, 32, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', same_padding=True),
            conv_bn_rel_dp(32, 32, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', same_padding=True),
            mp(kernel_size=2, dim=dim),
            conv_bn_rel_dp(32, 64, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', same_padding=True),
            conv_bn_rel_dp(64, 64, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', same_padding=True),
        )
        self.decoder = nn.Sequential(
            conv_bn_rel_dp(128, 64, kernel_size=2, stride=2, dim=dim, reverse=True, activate_unit='leaky_relu'),
            conv_bn_rel_dp(64, 64, kernel_size=3, stride=1, dim=dim, reverse=False, activate_unit='leaky_relu', same_padding=True),
            conv_bn_rel_dp(64, 32, kernel_size=2, stride=2, dim=dim, reverse=True, activate_unit='leaky_relu'),
            conv_bn_rel_dp(32, 32, kernel_size=3, stride=1, dim=dim, reverse=False, activate_unit='leaky_relu', same_padding=True),
            conv_bn_rel_dp(32, 16, kernel_size=2, stride=2, dim=dim, reverse=True, activate_unit='leaky_relu'),
            conv_bn_rel_dp(16, 8, kernel_size=3, stride=1, dim=dim, reverse=False, activate_unit='None', same_padding=True),
            conv_bn_rel_dp(8, 1, kernel_size=3, stride=1, dim=dim, reverse=False, activate_unit='None', same_padding=True)
        )
        return

    def encode(self, x):
        x = self.encoder(x)
        return self.conv11(x), self.conv12(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z


    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)

        return output, mu, logvar



