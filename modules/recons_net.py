from modules.layers import *
import torch
import torch.nn as nn
import numpy as np
import losses.loss as loss

import pyreg.module_parameters as pars
import pyreg.similarity_measure_factory as smf
from pyreg.external_variable import *


class ReconsNet(nn.Module):
    def __init__(self, model_config, network_mode):
        super(ReconsNet, self).__init__()
        self.config = model_config
        self.network_mode = network_mode
        print("ReconsNet Mode: {}".format(self.network_mode))

        use_bn = self.config['pregis_net']['recons_net']['bn']
        use_dp = self.config['pregis_net']['recons_net']['dp']
        dim = self.config['dim']
        self.img_sz = self.config['img_sz']
        self.batch_size = self.img_sz[0]
        self.last_conv_sz = (np.array(self.img_sz) / 16).astype(np.int)
        use_tv_loss = self.config['pregis_net']['recons_net']['use_TV_loss']
        self.L1_weight = self.config['pregis_net']['recons_net']['L1_weight']
        self.recons_criterion_L1 = nn.L1Loss(reduction='mean').cuda()
        self.KLD_weight = self.config['pregis_net']['recons_net']['KLD_weight']
        if use_tv_loss:
            self.recons_criterion_TV = loss.TVLoss().cuda()
            self.TV_weight = self.config['pregis_net']['recons_net']['TV_weight']
        else:
            self.recons_criterion_TV = None
        self.recons_weight = self.config['pregis_net']['recons_net']['recons_weight']
        self.sim_weight = self.config['pregis_net']['recons_net']['sim_weight']
        self.mode_change_epoch = self.config['pregis_net']['recons_net']['mode_change_epoch']

        # convolution to get mu and logvar
        self.conv11 = ConBnRelDp(16, 8, kernel_size=3, stride=1, dim=dim, activate_unit='None', same_padding=True)
        # self.conv12 = ConBnRelDp(16, 1, kernel_size=3, stride=1, dim=dim, activate_unit='None', same_padding=True)
        self.conv21 = ConBnRelDp(16, 8, kernel_size=3, stride=1, dim=dim, activate_unit='None', same_padding=True)
        # self.conv22 = ConBnRelDp(16, 1, kernel_size=3, stride=1, dim=dim, activate_unit='None', same_padding=True)

        # self.conv3 = ConBnRelDp(1, 16, kernel_size=3, stride=1, dim=dim, activate_unit='None', same_padding=True)
        self.conv4 = ConBnRelDp(8, 16, kernel_size=3, stride=1, dim=dim, activate_unit='None', same_padding=True)

        self.encoder_conv11 = ConBnRelDp(1, 8, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu',
                                         same_padding=True, use_bn=use_bn, use_dp=use_dp)
        self.encoder_conv12 = ConBnRelDp(1, 8, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu',
                                         same_padding=True, use_bn=use_bn, use_dp=use_dp)
        self.encoder_conv3 = ConBnRelDp(16, 32, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu',
                                        same_padding=True, use_bn=use_bn, use_dp=use_dp)
        self.encoder_conv4 = ConBnRelDp(32, 64, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu',
                                        same_padding=True, use_bn=use_bn, use_dp=use_dp)
        self.encoder_conv5 = ConBnRelDp(64, 32, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu',
                                        same_padding=True, use_bn=use_bn, use_dp=use_dp)
        self.encoder_conv6 = ConBnRelDp(32, 16, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu',
                                        same_padding=True, use_bn=use_bn, use_dp=use_dp)

        self.decoder_conv1 = ConBnRelDp(16, 32, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu',
                                        same_padding=True, use_bn=use_bn, use_dp=use_dp)
        self.decoder_conv2 = ConBnRelDp(32, 64, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu',
                                        same_padding=True, use_bn=use_bn, use_dp=use_dp)
        self.decoder_conv3 = ConBnRelDp(64, 32, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu',
                                        same_padding=True, use_bn=use_bn, use_dp=use_dp)
        self.decoder_conv4 = ConBnRelDp(32, 16, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu',
                                        same_padding=True, use_bn=use_bn, use_dp=use_dp)
        self.decoder_conv5 = ConBnRelDp(16, 8, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu',
                                        same_padding=True, use_bn=use_bn, use_dp=use_dp)
        self.decoder_conv6 = ConBnRelDp(8, 1, kernel_size=3, stride=1, dim=dim, activate_unit='None',
                                        same_padding=True, use_bn=use_bn, use_dp=use_dp)
        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU()
        self.max_pool = MaxPool(kernel_size=2, dim=dim, return_indieces=True)
        self.max_unpool = MaxUnpool(kernel_size=2, dim=dim)

        self.indices1 = None
        self.indices2 = None
        self.indices3 = None
        self.indices4 = None

        self.img_sz = self.config['img_sz']
        self.mermaid_config_file = model_config['mermaid_config_file']
        self.spacing = 1. / (np.array(self.img_sz[2:]) - 1)

        params = pars.ParameterDict()
        params.load_JSON(self.mermaid_config_file)
        sm_factory = smf.SimilarityMeasureFactory(self.spacing)
        self.sim_criterion = sm_factory.create_similarity_measure(params['model']['registration_model'])

        self.relu = nn.ReLU(inplace=True)
        # results
        self.mu = None
        self.log_var = None
        self.recons_image = None
        self.diff_image = None
        return

    def encode(self, x1, x2):
        x1 = self.encoder_conv11(x1)
        x2 = self.encoder_conv12(x2)
        # x = self.encoder_conv2(x)
        x = torch.cat((x1, x2), dim=1)
        x, self.indices1 = self.max_pool(x)
        x = self.encoder_conv3(x)
        x, self.indices2 = self.max_pool(x)
        x = self.encoder_conv4(x)
        x, self.indices3 = self.max_pool(x)
        x = self.encoder_conv5(x)
        x, self.indices4 = self.max_pool(x)
        x = self.encoder_conv6(x)
        # x = self.relu(self.fc1(x.view(-1, np.prod(self.last_conv_sz[2:]) * 1)))
        return self.conv11(x), self.conv21(x)

    def decode(self, z):
        # fc3 = self.relu(self.fc3(z))
        # x = self.relu(self.fc4(fc3)).view(-1, 1, *self.last_conv_sz[2:])
        x = self.conv4(z)
        x = self.decoder_conv1(x)
        x = self.max_unpool(x, self.indices4)
        x = self.decoder_conv2(x)
        x = self.max_unpool(x, self.indices3)
        x = self.decoder_conv3(x)
        x = self.max_unpool(x, self.indices2)
        x = self.decoder_conv4(x)
        x = self.max_unpool(x, self.indices1)
        x = self.decoder_conv5(x)
        x = self.decoder_conv6(x)
        return self.sigmoid(x)

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def calculate_vae_loss(self, input_image, target_image, current_epoch):
        # if current_epoch > self.mode_change_epoch:
        #    self.network_mode = 'pregis'
        loss_dict = {}
        kld_element = self.mu.pow(2).add_(self.log_var.exp()).mul_(-1).add_(1).add_(self.log_var)
        kld_loss = torch.mean(kld_element).mul_(-0.5)
        loss_dict['vae_kld_loss'] = kld_loss

        recons_loss_11 = self.recons_criterion_L1(input_image, self.recons_image)
        loss_dict['recons_loss_l1'] = recons_loss_11
        if self.recons_criterion_TV is not None:
            recons_loss_tv = self.recons_criterion_TV(self.recons_image, input_image)
            loss_dict['recons_loss_TV'] = recons_loss_tv
            recons_loss = self.L1_weight * recons_loss_11 + self.TV_weight * recons_loss_tv
        else:
            recons_loss = recons_loss_11
        loss_dict['vae_recons_loss'] = recons_loss

        all_vae_loss = self.KLD_weight * kld_loss + self.recons_weight * recons_loss
        # if self.network_mode == 'pregis':
        sim_loss = self.sim_criterion.compute_similarity_multiNC(self.recons_image, target_image) / self.batch_size
        loss_dict['vae_sim_loss'] = sim_loss
        sim_factor = 1./(np.exp((100-current_epoch)/20)+1)
        all_vae_loss += self.sim_weight * sim_factor * sim_loss

        loss_dict['vae_all_loss'] = all_vae_loss
        return loss_dict

    def forward(self, input_image, target_image):
        self.mu, self.log_var = self.encode(input_image, target_image)
        z = self.reparameterize(self.mu, self.log_var)
        # z = self.encode(input_image, target_image)
        self.recons_image = self.decode(z)
        diff_image = input_image - self.recons_image.detach()
        return self.recons_image, diff_image
