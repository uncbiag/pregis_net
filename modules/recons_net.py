from modules.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import losses.loss as loss

import pyreg.module_parameters as pars
import pyreg.model_factory as py_mf
import pyreg.utils as py_utils
import pyreg.image_sampling as py_is
import pyreg.similarity_measure_factory as smf
from pyreg.external_variable import *


class ReconsNet(nn.Module):
    def __init__(self, model_config):
        super(ReconsNet, self).__init__()

        self.use_bn = model_config['pregis_net']['bn']
        self.use_dp = model_config['pregis_net']['dp']
        self.dim = model_config['dim']
        # self.img_sz = model_config['img_sz']
        # self.batch_size = self.img_sz[0]

        self.recons_weight = model_config['pregis_net']['recons_weight']
        self.recons_criterion_L1 = nn.L1Loss(reduction='mean').cuda()

        self.__setup_network_structure()
        # results to return
        self.recons = None

        return

    def calculate_evaluation_loss(self, image, normal_mask):
        # same for training
        loss_dict = self.calculate_loss(image, normal_mask)
        # for recons net evaluation loss is reconstruction loss
        loss_dict['eval_loss'] = loss_dict['recons_loss']
        return loss_dict

    def calculate_loss(self, image, normal_mask):
        normal_indices = (normal_mask > 0.1)
        image_in_mask = image[normal_indices]
        recons_in_mask = self.recons[normal_indices]
        recons_loss = self.recons_criterion_L1(image_in_mask, recons_in_mask)
        all_loss = self.recons_weight * recons_loss
        loss_dict = {
            'recons_loss': recons_loss,
            'all_loss': all_loss
        }
        return loss_dict

    def forward(self, target_image):
        x = self.ec_1(target_image)
        x, indices_l1 = self.ec_2(x)
        x = self.ec_3(x)
        x = self.ec_4(x)
        x, indices_l2 = self.ec_5(x)
        x = self.ec_6(x)
        x = self.ec_7(x)
        x, indices_l3 = self.ec_8(x)
        x = self.ec_9(x)
        x = self.ec_10(x)
        x, indices_l4 = self.ec_11(x)
        z = self.ec_12(x)

        # Decode Brain
        x = self.dc_13(z)
        x = self.dc_14(x, indices_l4)
        x = self.dc_15(x)
        x = self.dc_16(x)
        x = self.dc_17(x, indices_l3)
        x = self.dc_18(x)
        x = self.dc_19(x)
        x = self.dc_20(x, indices_l2)
        x = self.dc_21(x)
        x = self.dc_22(x)
        x = self.dc_23(x, indices_l1)
        x = self.dc_24(x)
        x = self.dc_25(x)
        self.recons = self.dc_26(x)

        return

    def __setup_network_structure(self):
        self.ec_1 = ConBnRelDp(1, 16, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_2 = MaxPool(2, dim=self.dim, return_indieces=True)
        # self.ec_2 = ConBnRelDp(16, 16, kernel_size=3, stride=2, dim=self.dim, activate_unit='leaky_relu',
        #                        use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_3 = ConBnRelDp(16, 32, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_4 = ConBnRelDp(32, 32, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_5 = MaxPool(2, dim=self.dim, return_indieces=True)
        self.ec_6 = ConBnRelDp(32, 64, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_7 = ConBnRelDp(64, 64, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_8 = MaxPool(2, dim=self.dim, return_indieces=True)
        self.ec_9 = ConBnRelDp(64, 128, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_10 = ConBnRelDp(128, 128, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_11 = MaxPool(2, dim=self.dim, return_indieces=True)
        self.ec_12 = ConBnRelDp(128, 256, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)

        # Decoder for Brain
        self.dc_13 = ConBnRelDp(256, 128, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_14 = MaxUnpool(kernel_size=2, dim=self.dim)
        self.dc_15 = ConBnRelDp(128, 128, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_16 = ConBnRelDp(128, 64, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_17 = MaxUnpool(kernel_size=2, dim=self.dim)
        self.dc_18 = ConBnRelDp(64, 64, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_19 = ConBnRelDp(64, 32, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_20 = MaxUnpool(kernel_size=2, dim=self.dim)
        self.dc_21 = ConBnRelDp(32, 32, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_22 = ConBnRelDp(32, 16, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_23 = MaxUnpool(kernel_size=2, dim=self.dim)
        self.dc_24 = ConBnRelDp(16, 16, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_25 = ConBnRelDp(16, 8, kernel_size=3, stride=1, dim=self.dim, activate_unit='None')
        self.dc_26 = ConBnRelDp(8, 1, kernel_size=3, stride=1, dim=self.dim, activate_unit='None')
