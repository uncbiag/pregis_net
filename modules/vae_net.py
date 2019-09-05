from modules.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import losses.loss as loss

class VaeNet(nn.Module):
    def __init__(self, model_config):
        super(VaeNet, self).__init__()
        self.config = model_config

        use_bn = self.config['bn']
        use_dp = self.config['dp']
        dim = self.config['dim']

        model_name = 'pregis_net'
        use_TV_loss = self.config[model_name]['recons_net']['use_TV_loss']
        self.recons_criterion_L1 = nn.L1Loss(reduction='mean').cuda()

        self.KLD_weight = self.config[model_name]['recons_net']['KLD_weight']
        if use_TV_loss:
            self.recons_criterion_TV = loss.TVLoss().cuda()
            self.TV_weight = self.config[model_name]['recons_net']['TV_weight']
        else:
            self.recons_criterion_TV = None

        #convolution to get mu and logvar
        self.conv11 = conv_bn_rel_dp(64, 128, kernel_size=3, stride=2, dim=dim, activate_unit='None', same_padding=True)
        self.conv12 = conv_bn_rel_dp(64, 128, kernel_size=3, stride=2, dim=dim, activate_unit='None', same_padding=True)

        self.encoder = nn.Sequential(
            conv_bn_rel_dp(1, 8, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', same_padding=True, use_bn=use_bn,use_dp=use_dp),
            conv_bn_rel_dp(8, 16, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', same_padding=True, use_bn=use_bn, use_dp=use_dp),
            mp(kernel_size=2, dim=dim),
            conv_bn_rel_dp(16, 32, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', same_padding=True, use_bn=use_bn, use_dp=use_dp),
            conv_bn_rel_dp(32, 32, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', same_padding=True, use_bn=use_bn, use_dp=use_dp),
            mp(kernel_size=2, dim=dim),
            conv_bn_rel_dp(32, 64, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', same_padding=True, use_bn=use_bn, use_dp=use_dp),
            conv_bn_rel_dp(64, 64, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', same_padding=True, use_bn=use_bn, use_dp=use_dp),
        )
        self.decoder = nn.Sequential(
            conv_bn_rel_dp(128, 64, kernel_size=2, stride=2, dim=dim, reverse=True, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp),
            conv_bn_rel_dp(64, 64, kernel_size=3, stride=1, dim=dim, reverse=False, activate_unit='leaky_relu', same_padding=True, use_bn=use_bn, use_dp=use_dp),
            conv_bn_rel_dp(64, 32, kernel_size=2, stride=2, dim=dim, reverse=True, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp),
            conv_bn_rel_dp(32, 32, kernel_size=3, stride=1, dim=dim, reverse=False, activate_unit='leaky_relu', same_padding=True, use_bn=use_bn, use_dp=use_dp),
            conv_bn_rel_dp(32, 16, kernel_size=2, stride=2, dim=dim, reverse=True, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp),
            conv_bn_rel_dp(16, 8, kernel_size=3, stride=1, dim=dim, reverse=False, activate_unit='None', same_padding=True),
            conv_bn_rel_dp(8, 1, kernel_size=3, stride=1, dim=dim, reverse=False, activate_unit='None', same_padding=True),
            nn.Sigmoid()
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

    def calculate_vae_loss(self, moving_image):

        loss_dict = {}
        kld_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
        kld_loss = torch.mean(kld_element).mul_(-0.5)
        loss_dict['vae_kld_loss'] = kld_loss

        recons_loss_L1 = self.recons_criterion_L1(self.recons_image, moving_image)
        if self.recons_criterion_TV is not None:
            recons_loss_TV = self.recons_criterion_TV(self.recons_image, moving_image)
            loss_dict['recons_loss_TV'] = recons_loss_TV
            recons_loss = recons_loss_L1 + self.TV_weight * recons_loss_TV
        else:
            recons_loss = recons_loss_L1

        all_vae_loss = self.KLD_weight * kld_loss + recons_loss
        loss_dict['vae_all_loss'] = all_vae_loss
        loss_dict['vae_recons_loss'] = recons_loss

        return loss_dict



    def forward(self, input):
        self.mu, self.logvar = self.encode(input)
        z = self.reparameterize(self.mu, self.logvar)
        self.recons_image = self.decode(z)

        return self.recons_image



