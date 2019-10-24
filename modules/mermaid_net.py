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
        self.l1_loss = nn.L1Loss(reduction='mean').cuda()

        self.mermaid_config_file = model_config['mermaid_config_file']
        self.mermaid_unit = None
        self.spacing = 1. / (np.array(self.img_sz[2:]) - 1)

        # members that will be set during mermaid initialization
        self.similarity_measure_type = None
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

        # results to return
        self.momentum = None
        self.warped_image = None
        self.phi = None

        return

    def init_mermaid_env(self, spacing):
        params = pars.ParameterDict()
        params.load_JSON(self.mermaid_config_file)
        # sm_factory = smf.SimilarityMeasureFactory(self.spacing)
        # self.sim_criterion = sm_factory.create_similarity_measure(params['model']['registration_model'])
        model_name = params['model']['registration_model']['type']
        self.similarity_measure_type = params['model']['registration_model']['similarity_measure']['type']
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

        # SVFVectorMomentumMapNet, LDDMMShootingVectorMomentumMapNet
        self.mermaid_unit = model.cuda()
        self.mermaid_criterion = criterion
        self.sampler = py_is.ResampleImage()
        return

    def calculate_evaluation_loss(self, moving, target, normal_mask, disp_field):
        # same for training
        loss_dict = self.calculate_loss(moving, target, normal_mask)
        far_indices = (normal_mask > 1.5)
        near_indices = ((normal_mask < 1.5) & (normal_mask > 0.5))
        tumor_indices = (normal_mask < 0.5)
        model_disp_field = self.phi - self.identityMap
        for dim in range(self.dim):
            model_disp_field[:, dim, ...] = model_disp_field[:, dim, ...] / self.spacing[dim]

        disp_diff = torch.norm((model_disp_field - disp_field), p=2, dim=1, keepdim=True)

        tumor_disp_loss = torch.mean(disp_diff[tumor_indices])
        near_disp_loss = torch.mean(disp_diff[near_indices])
        far_disp_loss = torch.mean(disp_diff[far_indices])

        loss_dict['tumor_disp_loss'] = tumor_disp_loss
        loss_dict['near_disp_loss'] = near_disp_loss
        loss_dict['far_disp_loss'] = far_disp_loss
        loss_dict['eval_loss'] = tumor_disp_loss + near_disp_loss + 0.5 * far_disp_loss
        return loss_dict

    def calculate_loss(self, moving, target, normal_mask):
        if self.similarity_measure_type == 'maskncc':
            target = torch.cat((target, normal_mask), dim=1)
        mermaid_all_loss, mermaid_sim_loss, mermaid_reg_loss = self.mermaid_criterion(
            phi0=self.identityMap,
            phi1=self.phi,
            I0_source=moving,
            I1_target=target,
            lowres_I0=None,
            variables_from_forward_model=self.mermaid_unit.get_variables_to_transfer_to_loss_function(),
            variables_from_optimizer=None
        )
        loss_dict = {
            'all_loss': mermaid_all_loss / self.batch_size,
            'mermaid_all_loss': mermaid_all_loss / self.batch_size,
            'mermaid_sim_loss': mermaid_sim_loss / self.batch_size,
            'mermaid_reg_loss': mermaid_reg_loss / self.batch_size
        }
        loss_dict['all_loss'] = loss_dict['mermaid_all_loss']

        return loss_dict

    def forward(self, input_image, target_image):
        x1 = self.ec_1(input_image)
        x2 = self.ec_2(target_image)
        x_l1 = torch.cat((x1, x2), dim=1)
        x = self.ec_3(x_l1)
        x = self.ec_4(x)
        x_l2 = self.ec_5(x)
        x = self.ec_6(x_l2)
        x = self.ec_7(x)
        x_l3 = self.ec_8(x)
        x = self.ec_9(x_l3)
        x = self.ec_10(x)
        x_l4 = self.ec_11(x)
        x = self.ec_12(x_l4)
        z = self.ec_13(x)

        # Decode Momentum
        x = self.dc_14(z)
        x = torch.cat((x_l4, x), dim=1)
        x = self.dc_15(x)
        x = self.dc_16(x)
        x = self.dc_17(x)
        x = torch.cat((x_l3, x), dim=1)
        x = self.dc_18(x)
        x = self.dc_19(x)
        x = self.dc_20(x)
        x = torch.cat((x_l2, x), dim=1)
        x = self.dc_21(x)
        x = self.dc_22(x)
        self.momentum = self.dc_23(x)

        warped_image, phi = self.mermaid_shoot(input_image, target_image, self.momentum)
        self.warped_image = warped_image
        self.phi = phi
        return

    def set_mermaid_params(self, moving, target, momentum):
        self.mermaid_unit.set_dictionary_to_pass_to_integrator({'I0': moving, 'I1': target})
        self.mermaid_criterion.set_dictionary_to_pass_to_smoother({'I0': moving, 'I1': target})
        self.mermaid_unit.m = momentum
        self.mermaid_criterion.m = momentum
        return

    def mermaid_shoot(self, moving, target, momentum):
        self.set_mermaid_params(moving=moving, target=target, momentum=momentum)
        low_res_phi = self.mermaid_unit(self.lowResIdentityMap)
        desired_sz = self.identityMap.size()[2:]
        phi, _ = self.sampler.upsample_image_to_size(low_res_phi, self.lowResSpacing, desired_sz, spline_order=1)
        moving_warped = py_utils.compute_warped_image_multiNC(moving, phi, self.spacing, spline_order=1,
                                                              zero_boundary=True)
        return moving_warped, phi

    def __setup_network_structure(self):
        self.ec_1 = ConBnRelDp(1, 8, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_2 = ConBnRelDp(1, 8, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_3 = ConBnRelDp(16, 16, kernel_size=3, stride=2, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_4 = ConBnRelDp(16, 32, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_5 = ConBnRelDp(32, 32, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_6 = MaxPool(2, dim=self.dim)
        self.ec_7 = ConBnRelDp(32, 64, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_8 = ConBnRelDp(64, 64, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_9 = MaxPool(2, dim=self.dim)
        self.ec_10 = ConBnRelDp(64, 128, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_11 = ConBnRelDp(128, 128, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.ec_12 = MaxPool(2, dim=self.dim)
        self.ec_13 = ConBnRelDp(128, 256, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)

        # decoder for momentum
        self.dc_14 = ConBnRelDp(256, 128, kernel_size=2, stride=2, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp, reverse=True)
        self.dc_15 = ConBnRelDp(256, 128, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_16 = ConBnRelDp(128, 128, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_17 = ConBnRelDp(128, 64, kernel_size=2, stride=2, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp, reverse=True)
        self.dc_18 = ConBnRelDp(128, 64, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_19 = ConBnRelDp(64, 64, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_20 = ConBnRelDp(64, 32, kernel_size=2, stride=2, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp, reverse=True)
        self.dc_21 = ConBnRelDp(64, 32, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                use_bn=self.use_bn, use_dp=self.use_dp)
        self.dc_22 = ConBnRelDp(32, 16, kernel_size=3, stride=1, dim=self.dim, activate_unit='None')
        self.dc_23 = ConBnRelDp(16, self.dim, kernel_size=3, stride=1, dim=self.dim, activate_unit='None')
