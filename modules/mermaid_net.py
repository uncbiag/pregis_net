from modules.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pyreg.module_parameters as pars
import pyreg.model_factory as py_mf
import pyreg.utils as py_utils
import pyreg.image_sampling as py_is
import pyreg.similarity_measure_factory as smf
from pyreg.external_variable import *
from utils.registration_method import _get_low_res_size_from_size, _get_low_res_spacing_from_spacing, \
    _compute_low_res_image


class MermaidNet(nn.Module):
    def __init__(self, model_config):
        super(MermaidNet, self).__init__()

        self.use_bn = model_config['pregis_net']['bn']
        self.use_dp = model_config['pregis_net']['dp']
        self.dim = model_config['dim']
        self.img_sz = model_config['img_sz']
        self.batch_size = self.img_sz[0]

        self.recons_weight = model_config['pregis_net']['recons_weight']
        self.recons_criterion_L1 = nn.L1Loss(reduction='mean').cuda()
        self.segmentation_weight = model_config['pregis_net']['seg_weight']
        self.segmentation_criterion = nn.BCELoss(reduction='mean').cuda()

        self.mermaid_config_file = model_config['mermaid_config_file']
        self.mermaid_unit = None
        self.spacing = 1. / (np.array(self.img_sz[2:]) - 1)

        # members that will be set during mermaid initialization
        self.use_map = None
        self.map_low_res_factor = None
        self.lowResSize = None
        self.lowResSpacing = None
        self.identityMap = None
        self.lowResIdentityMap = None
        self.mermaid_criterion = None
        self.sampler = None

        self.init_mermaid_env(spacing=self.spacing)
        self.__setup_network_structure()

        self.moving_image = None
        self.moving_label1 = None
        self.moving_label2 = None

        self.target_image = None
        self.target_label1 = None
        self.target_label2 = None

        self.weighted_mask = None
        # results to return
        self.momentum = None
        self.moving_image_and_label = None
        self.warped_image_and_label = None
        self.target_image_and_label = None
        self.phi = None

        return

    def init_mermaid_env(self, spacing):
        params = pars.ParameterDict()
        params.load_JSON(self.mermaid_config_file)
        sm_factory = smf.SimilarityMeasureFactory(self.spacing)
        self.sim_criterion = sm_factory.create_similarity_measure(params['model']['registration_model'])
        model_name = params['model']['registration_model']['type']
        self.use_map = params['model']['deformation']['use_map']
        self.map_low_res_factor = params['model']['deformation'][('map_low_res_factor', None, 'low_res_factor')]
        compute_similarity_measure_at_low_res = params['model']['deformation'][
            ('compute_similarity_measure_at_low_res', False, 'to compute Sim at lower resolution')]

        # Currently Must use map_low_res_factor = 0.5
        if self.map_low_res_factor is not None:
            self.lowResSize = _get_low_res_size_from_size(self.img_sz, self.map_low_res_factor)
            self.lowResSpacing = _get_low_res_spacing_from_spacing(spacing, self.img_sz, self.lowResSize)
            if compute_similarity_measure_at_low_res:
                mf = py_mf.ModelFactory(self.lowResSize, self.lowResSpacing, self.lowResSize, self.lowResSpacing)
            else:
                mf = py_mf.ModelFactory(self.img_sz, self.spacing, self.lowResSize, self.lowResSpacing)
        else:
            raise ValueError("map_low_res_factor not defined")
        model, criterion = mf.create_registration_model(model_name, params['model'], compute_inverse_map=False)

        # Currently Must use map
        if self.use_map:
            # create the identity map [0,1]^d, since we will use a map-based implementation
            _id = py_utils.identity_map_multiN(self.img_sz, spacing)
            self.identityMap = torch.from_numpy(_id).cuda()
            if self.map_low_res_factor is not None:
                # create a lower resolution map for the computations
                lowres_id = py_utils.identity_map_multiN(self.lowResSize, self.lowResSpacing)
                self.lowResIdentityMap = torch.from_numpy(lowres_id).cuda()

        # SVFVectorMometumMapNet, LDDMMShootingVectorMomentumMapNet
        self.mermaid_unit = model.cuda()
        self.mermaid_criterion = criterion
        self.sampler = py_is.ResampleImage()
        return

    def calculate_pregis_loss(self):
        self.target_image_and_label = self.target_image.cuda()
        if self.target_label1 is not None:
            self.target_image_and_label = torch.cat((self.target_image_and_label, self.target_label1.cuda()), dim=1)
        if self.target_label2 is not None:
            self.target_image_and_label = torch.cat((self.target_image_and_label, self.target_label2.cuda()), dim=1)
        target_image_and_label = torch.cat((self.target_image_and_label, self.weighted_mask.cuda()), dim=1)

        self.moving_image_and_label = torch.cat((self.moving_image.cuda(),
                                                 self.moving_label1.cuda(),
                                                 self.moving_label2.cuda()),
                                                dim=1)
        mermaid_all_loss, mermaid_sim_loss, mermaid_reg_loss = self.mermaid_criterion(
            phi0=self.identityMap,
            phi1=self.phi,
            I0_source=self.moving_image_and_label,
            I1_target=target_image_and_label,
            lowres_I0=None,
            variables_from_forward_model=self.mermaid_unit.get_variables_to_transfer_to_loss_function(),
            variables_from_optimizer=None
        )
        loss_dict = {
            'mermaid_all_loss': mermaid_all_loss / self.batch_size,
            'mermaid_sim_loss': mermaid_sim_loss / self.batch_size,
            'mermaid_reg_loss': mermaid_reg_loss / self.batch_size
        }

        return loss_dict

    def forward(self, image_dict):
        self.moving_image = image_dict['CT_image']
        self.moving_label1 = image_dict['CT_SmLabel']
        self.moving_label2 = image_dict['CT_SdLabel']

        self.target_image = image_dict['CBCT_image']
        if 'CBCT_SmLabel' in image_dict and 'CBCT_SdLabel' in image_dict:
            self.target_label1 = image_dict['CBCT_SmLabel']
            self.target_label2 = image_dict['CBCT_SdLabel']

        self.weighted_mask = image_dict['weighted_mask']

        x1 = self.encoder_conv_1i(self.moving_image.cuda())
        x2 = self.encoder_conv_2i(self.target_image.cuda())
        x_l1 = torch.cat((x1, x2), dim=1)
        x = self.encoder_conv_3d(x_l1)
        x = self.encoder_conv_4(x)
        x_l2 = self.encoder_conv_5(x)
        x, indices_l2 = self.encoder_maxpool_6(x_l2)
        x = self.encoder_conv_7(x)
        x_l3 = self.encoder_conv_8(x)
        x, indices_l3 = self.encoder_maxpool_9(x_l3)
        x = self.encoder_conv_10(x)
        x_l4 = self.encoder_conv_11(x)
        x, indices_l4 = self.encoder_maxpool_12(x_l4)
        z = self.encoder_conv_13(x)

        # Decode Momentum
        x = self.decoder1_conv_14u(z)
        x = torch.cat((x_l4, x), dim=1)
        x = self.decoder1_conv_15(x)
        x = self.decoder1_conv_16(x)
        x = self.decoder1_conv_17u(x)
        x = torch.cat((x_l3, x), dim=1)
        x = self.decoder1_conv_18(x)
        x = self.decoder1_conv_19(x)
        x = self.decoder1_conv_20u(x)
        x = torch.cat((x_l2, x), dim=1)
        x = self.decoder1_conv_21(x)
        self.momentum = self.decoder1_conv_22o(x)
        del x, x1, x2, x_l1, x_l2, x_l3, x_l4

        moving_labels = torch.cat((self.moving_label1.cuda(), self.moving_label2.cuda()), dim=1)
        self.mermaid_shoot(self.moving_image.cuda(), moving_labels, self.momentum)

        return

    def set_mermaid_params(self, momentum):
        # self.mermaid_unit.set_dictionary_to_pass_to_integrator({'I0': moving, 'I1': target})
        # self.mermaid_criterion.set_dictionary_to_pass_to_smoother({'I0': moving, 'I1': target})
        self.mermaid_unit.m = momentum
        self.mermaid_criterion.m = momentum
        return

    def mermaid_shoot(self, moving_image, moving_labels, momentum):
        self.set_mermaid_params(momentum)
        low_res_phi = self.mermaid_unit(self.lowResIdentityMap)
        desired_sz = self.identityMap.size()[2:]
        self.phi, _ = self.sampler.upsample_image_to_size(low_res_phi, self.lowResSpacing, desired_sz, spline_order=1)
        warped_image = py_utils.compute_warped_image_multiNC(moving_image, self.phi, self.spacing, spline_order=1,
                                                             zero_boundary=True)
        warped_label = py_utils.compute_warped_image_multiNC(moving_labels, self.phi, self.spacing, spline_order=0,
                                                             zero_boundary=True)
        self.warped_image_and_label = torch.cat((warped_image, warped_label), dim=1)
        return

    def __setup_network_structure(self):
        self.encoder_conv_1i = ConBnRelDp(1, 8, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                          use_bn=self.use_bn, use_dp=self.use_dp)
        self.encoder_conv_2i = ConBnRelDp(1, 8, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                          use_bn=self.use_bn, use_dp=self.use_dp)
        self.encoder_conv_3d = ConBnRelDp(16, 16, kernel_size=3, stride=2, dim=self.dim, activate_unit='leaky_relu',
                                          use_bn=self.use_bn, use_dp=self.use_dp)
        self.encoder_conv_4 = ConBnRelDp(16, 32, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                         use_bn=self.use_bn, use_dp=self.use_dp)
        self.encoder_conv_5 = ConBnRelDp(32, 32, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                         use_bn=self.use_bn, use_dp=self.use_dp)
        self.encoder_maxpool_6 = MaxPool(2, dim=self.dim, return_indieces=True)
        self.encoder_conv_7 = ConBnRelDp(32, 64, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                         use_bn=self.use_bn, use_dp=self.use_dp)
        self.encoder_conv_8 = ConBnRelDp(64, 64, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                         use_bn=self.use_bn, use_dp=self.use_dp)
        self.encoder_maxpool_9 = MaxPool(2, dim=self.dim, return_indieces=True)
        self.encoder_conv_10 = ConBnRelDp(64, 128, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                          use_bn=self.use_bn, use_dp=self.use_dp)
        self.encoder_conv_11 = ConBnRelDp(128, 128, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                          use_bn=self.use_bn, use_dp=self.use_dp)
        self.encoder_maxpool_12 = MaxPool(2, dim=self.dim, return_indieces=True)
        self.encoder_conv_13 = ConBnRelDp(128, 256, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                          use_bn=self.use_bn, use_dp=self.use_dp)

        # Decoder for momentum
        self.decoder1_conv_14u = ConBnRelDp(256, 128, kernel_size=2, stride=2, dim=self.dim, activate_unit='leaky_relu',
                                            use_bn=self.use_bn, use_dp=self.use_dp, reverse=True)
        self.decoder1_conv_15 = ConBnRelDp(256, 128, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                           use_bn=self.use_bn, use_dp=self.use_dp)
        self.decoder1_conv_16 = ConBnRelDp(128, 128, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                           use_bn=self.use_bn, use_dp=self.use_dp)
        self.decoder1_conv_17u = ConBnRelDp(128, 64, kernel_size=2, stride=2, dim=self.dim, activate_unit='leaky_relu',
                                            use_bn=self.use_bn, use_dp=self.use_dp, reverse=True)
        self.decoder1_conv_18 = ConBnRelDp(128, 64, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                           use_bn=self.use_bn, use_dp=self.use_dp)
        self.decoder1_conv_19 = ConBnRelDp(64, 64, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                           use_bn=self.use_bn, use_dp=self.use_dp)
        self.decoder1_conv_20u = ConBnRelDp(64, 32, kernel_size=2, stride=2, dim=self.dim, activate_unit='leaky_relu',
                                            use_bn=self.use_bn, use_dp=self.use_dp, reverse=True)
        self.decoder1_conv_21 = ConBnRelDp(64, 16, kernel_size=3, stride=1, dim=self.dim, activate_unit='None')
        self.decoder1_conv_22o = ConBnRelDp(16, self.dim, kernel_size=3, stride=1, dim=self.dim, activate_unit='None')
