from modules.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../mermaid'))
import pyreg.module_parameters as pars
import pyreg.model_factory as py_mf
import pyreg.utils as py_utils
import pyreg.similarity_measure_factory as smf
from pyreg.external_variable import *
from utils.registration_method import _get_low_res_size_from_size, _get_low_res_spacing_from_spacing, \
     get_resampled_image
from functools import partial


class MermaidNet(nn.Module):
    def __init__(self, model_config):
        super(MermaidNet, self).__init__()
        self.use_ct_labels_as_input = True
        self.use_bn = model_config['pregis_net']['bn']
        print("Use Batch Normalization: {}".format(self.use_bn))
        self.use_dp = model_config['pregis_net']['dp']
        self.dim = model_config['dim']
        self.img_sz = model_config['img_sz']
        self.batch_size = self.img_sz[0]

        self.step = 1
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
        self.lowRes_fn = None
        self.sim_criterion = None

        self.init_mermaid_env(spacing=self.spacing)
        self.__setup_network_structure__()

        # results to return
        self.phi = None
        self.loss_dict = None
        self.warped_image = None
        self.warped_labels = None

        return

    def init_mermaid_env(self, spacing):
        params = pars.ParameterDict()
        params.load_JSON(self.mermaid_config_file)
        sm_factory = smf.SimilarityMeasureFactory(self.spacing)
        self.sim_criterion = sm_factory.create_similarity_measure(params['model']['registration_model'])
        model_name = params['model']['registration_model']['type']
        self.use_map = params['model']['deformation']['use_map']
        self.map_low_res_factor = params['model']['deformation'][('map_low_res_factor', None, 'low_res_factor')]
        assert self.map_low_res_factor == 0.5
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
        print("Spacing: {}".format(self.spacing))
        print("LowResSize: {}".format(self.lowResSize))
        print("LowResIdentityMap Shape: {}".format(self.lowResIdentityMap.shape))
        self.lowRes_fn = partial(get_resampled_image, spacing=self.spacing, desired_size=self.lowResSize,
                                 zero_boundary=False, identity_map=self.lowResIdentityMap)
        return

    def __calculate_dice_socre(self, predicted_label, target_label, mask):
        roi_indices = (mask > 0.5)
        predicted_in_mask = predicted_label[roi_indices]
        target_in_mask = target_label[roi_indices]
        intersection = (predicted_in_mask * target_in_mask) .sum()
        smooth = 1.
        dice = (2 * intersection + smooth) / (predicted_in_mask.sum() + target_in_mask.sum() + smooth)
        return dice

    def __calculate_dice_score_multiN(self, predicted_labels, target_labels, mask):
        sm_label_dice = 0.0
        sd_label_dice = 0.0
        for batch in range(self.batch_size):
            sm_label_dice += self.__calculate_dice_socre(predicted_labels[batch, 0, ...],
                                                         target_labels[batch, 0, ...],
                                                         mask[batch, 0, ...])
            sd_label_dice += self.__calculate_dice_socre(predicted_labels[batch, 1, ...],
                                                         target_labels[batch, 1, ...],
                                                         mask[batch, 0, ...])
        return sm_label_dice, sd_label_dice

    def calculate_train_loss(self, moving_image_and_label, target_image_and_label):
        mermaid_all_loss, mermaid_sim_loss, mermaid_reg_loss = self.mermaid_criterion(
            phi0=self.identityMap,
            phi1=self.phi,
            I0_source=moving_image_and_label,
            I1_target=target_image_and_label,
            lowres_I0=None,
            variables_from_forward_model=self.mermaid_unit.get_variables_to_transfer_to_loss_function(),
            variables_from_optimizer=None
        )

        return mermaid_all_loss, mermaid_sim_loss, mermaid_reg_loss

    def forward(self, ct_image, cb_image, roi_label, ct_sblabel, ct_sdlabel, cb_sblabel, cb_sdlabel):

        # cb_image: [-1, 1], cb_image_n: [0, 1]
        # ct: same
        cb_image_n = (cb_image + 1) / 2.
        ct_image_n = (ct_image + 1) / 2.

        ct_labels = torch.cat((ct_sblabel, ct_sdlabel), dim=1)
        cb_labels = torch.cat((cb_sblabel, cb_sdlabel), dim=1)

        # moving: CBCT, target: CT
        cb_image_n_and_label = cb_image_n
        cb_image_n_and_label = torch.cat((cb_image_n_and_label, cb_sblabel), dim=1)
        cb_image_n_and_label = torch.cat((cb_image_n_and_label, cb_sdlabel), dim=1)

        ct_image_and_label = ct_image
        ct_image_and_label = torch.cat((ct_image_and_label, ct_sblabel), dim=1)
        ct_image_and_label = torch.cat((ct_image_and_label, ct_sdlabel), dim=1)
        ct_image_and_label = torch.cat((ct_image_and_label, roi_label), dim=1)

        ct_image_n_and_label = ct_image_n
        ct_image_n_and_label = torch.cat((ct_image_n_and_label, ct_sblabel), dim=1)
        ct_image_n_and_label = torch.cat((ct_image_n_and_label, ct_sdlabel), dim=1)
        ct_image_n_and_label = torch.cat((ct_image_n_and_label, roi_label), dim=1)

        # First step of multi-step network:
        # moving image is the warped image
        # identity map is the init phi

        warped_image = cb_image
        init_map = self.identityMap

        self.loss_dict = {
            'all_loss':  0.,
            'mermaid_all_loss': 0.,
            'mermaid_sim_loss': 0.,
            'mermaid_reg_loss': 0.,
            'dice_SmLabel': 0.,
            'dice_SdLabel': 0.,
        }
        out_roi = (F.interpolate(roi_label, scale_factor=0.5, mode='nearest') < 0.5)
        for i in range(self.step):
            # momentum is obtained from network (warped image to target image)
            if self.use_ct_labels_as_input:
                momentum = self.__network_forward__(moving_image=warped_image, target_image_and_label=ct_image_and_label)
            else:
                momentum = self.__network_forward__(moving_image=warped_image, target_image_and_label=ct_image)
            momentum.masked_fill_(out_roi, 0)
            # warped image is obtained by resampling the moving image
            warped_image_n, warped_labels, phi = self.__mermaid_shoot__(moving_image=cb_image_n, moving_labels=cb_labels,
                                                                        momentum=momentum, init_map=init_map)
            init_map = phi
            self.phi = phi

            # warped_image_n : [0, 1], warped_image: [-1, 1]
            warped_image = warped_image_n * 2 - 1
            self.warped_image = warped_image
            self.warped_labels = warped_labels
            all_loss, sim_loss, reg_loss = self.calculate_train_loss(moving_image_and_label=cb_image_n_and_label,
                                                                     target_image_and_label=ct_image_n_and_label)
            self.loss_dict['mermaid_all_loss'] += all_loss / self.batch_size
            self.loss_dict['mermaid_sim_loss'] += sim_loss / self.batch_size
            self.loss_dict['mermaid_reg_loss'] += reg_loss / self.batch_size

        self.loss_dict['all_loss'] = self.loss_dict['mermaid_all_loss']

        sm_label_dice, sd_label_dice = self.__calculate_dice_score_multiN(ct_labels,
                                                                          self.warped_labels.detach(),
                                                                          roi_label)
        self.loss_dict['dice_SmLabel'] = sm_label_dice / self.batch_size
        self.loss_dict['dice_SdLabel'] = sd_label_dice / self.batch_size

        return

    def __mermaid_shoot__(self, moving_image, moving_labels, momentum, init_map):
        # obtain transformation map from momentum
        # warp image and labels
        self.mermaid_unit.m = momentum
        self.mermaid_criterion.m = momentum
        low_res_phi = self.mermaid_unit(self.lowRes_fn(init_map))
        desired_sz = self.identityMap.size()[2:]
        phi = get_resampled_image(low_res_phi,
                                  self.lowResSpacing,
                                  desired_sz, 1, zero_boundary=False, identity_map=self.identityMap)

        warped_image = py_utils.compute_warped_image_multiNC(moving_image, phi, self.spacing, spline_order=1,
                                                             zero_boundary=True)
        warped_labels = py_utils.compute_warped_image_multiNC(moving_labels, phi, self.spacing, spline_order=0,
                                                              zero_boundary=True)
        return warped_image, warped_labels, phi

    def __network_forward__(self, moving_image, target_image_and_label):
        x1 = self.ec_1(moving_image)
        x2 = self.ec_2(target_image_and_label)
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
        output = self.dc_23(x)
        return output

    def __setup_network_structure__(self):
        self.ec_1 = ConBnRelDp(1, 8, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                               use_bn=self.use_bn, use_dp=self.use_dp)
        if self.use_ct_labels_as_input:
            self.ec_2 = ConBnRelDp(4, 8, kernel_size=3, stride=1, dim=self.dim, activate_unit='leaky_relu',
                                   use_bn=self.use_bn, use_dp=self.use_dp)
        else:
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

    def __network_forward_old__(self, moving_image, target_image_and_label):
        d1_1 = self.down_path_1_1(moving_image)
        d1_2 = self.down_path_1_2(target_image_and_label)
        d1 = torch.cat((d1_1, d1_2), 1)
        d2_1 = self.down_path_2_1(d1)
        d2_2 = self.down_path_2_2(d2_1)
        d2_2 = d2_1 + d2_2
        d2_3 = self.down_path_2_3(d2_2)
        d2_3 = d2_1 + d2_3
        d4_1 = self.down_path_4_1(d2_3)
        d4_2 = self.down_path_4_2(d4_1)
        d4_2 = d4_1 + d4_2
        d4_3 = self.down_path_4_3(d4_2)
        d4_3 = d4_2 + d4_3
        d8_1 = self.down_path_8_1(d4_3)
        d8_2 = self.down_path_8_2(d8_1)
        d8_2 = d8_1 + d8_2
        d8_3 = self.down_path_8_3(d8_2)
        d8_3 = d8_2 + d8_3
        d16_1 = self.down_path_16_1(d8_3)
        d16_2 = self.down_path_16_2(d16_1)
        d16_2 = d16_1 + d16_2

        u8_1 = self.up_path_8_1(d16_2)
        u8_2 = self.up_path_8_2(torch.cat((d8_3, u8_1), 1))
        u8_3 = self.up_path_8_3(u8_2)
        u8_3 = u8_2 + u8_3
        u4_1 = self.up_path_4_1(u8_3)
        u4_2 = self.up_path_4_2(torch.cat((d4_3, u4_1), 1))
        u4_3 = self.up_path_4_3(u4_2)
        u4_3 = u4_2 + u4_3
        u2_1 = self.up_path_2_1(u4_3)
        u2_2 = self.up_path_2_2(torch.cat((d2_3, u2_1), 1))
        output = self.up_path_2_3(u2_2)

        return output

    def __setup_network_structure_old__(self):
        self.down_path_1_1 = ConBnRelDp(1, 8, 3, stride=1, activate_unit='relu', same_padding=True, use_bn=False)
        if self.use_ct_labels_as_input:
            self.down_path_1_2 = ConBnRelDp(4, 8, 3, stride=1, activate_unit='relu', same_padding=True, use_bn=False)
        else:
            self.down_path_1_2 = ConBnRelDp(1, 8, 3, stride=1, activate_unit='relu', same_padding=True, use_bn=False)
        self.down_path_2_1 = ConBnRelDp(16, 32, 3, stride=2, activate_unit='relu', same_padding=True, use_bn=False)
        self.down_path_2_2 = ConBnRelDp(32, 32, 3, stride=1, activate_unit='relu', same_padding=True, use_bn=False)
        self.down_path_2_3 = ConBnRelDp(32, 32, 3, stride=1, activate_unit='relu', same_padding=True, use_bn=self.use_bn)
        self.down_path_4_1 = ConBnRelDp(32, 64, 3, stride=2, activate_unit='relu', same_padding=True, use_bn=self.use_bn)
        self.down_path_4_2 = ConBnRelDp(64, 64, 3, stride=1, activate_unit='relu', same_padding=True, use_bn=self.use_bn)
        self.down_path_4_3 = ConBnRelDp(64, 64, 3, stride=1, activate_unit='relu', same_padding=True, use_bn=self.use_bn)
        self.down_path_8_1 = ConBnRelDp(64, 128, 3, stride=2, activate_unit='relu', same_padding=True, use_bn=self.use_bn)
        self.down_path_8_2 = ConBnRelDp(128, 128, 3, stride=1, activate_unit='relu', same_padding=True, use_bn=self.use_bn)
        self.down_path_8_3 = ConBnRelDp(128, 128, 3, stride=1, activate_unit='relu', same_padding=True, use_bn=self.use_bn)
        self.down_path_16_1 = ConBnRelDp(128, 256, 3, stride=2, activate_unit='relu', same_padding=True, use_bn=self.use_bn)
        self.down_path_16_2 = ConBnRelDp(256, 256, 3, stride=1, activate_unit='relu', same_padding=True, use_bn=self.use_bn)

        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = ConBnRelDp(256, 128, 2, stride=2, activate_unit='leaky_relu', same_padding=False, use_bn=self.use_bn,
                                       reverse=True)
        self.up_path_8_2 = ConBnRelDp(128 + 128, 128, 3, stride=1, activate_unit='leaky_relu', same_padding=True, use_bn=self.use_bn)
        self.up_path_8_3 = ConBnRelDp(128, 128, 3, stride=1, activate_unit='leaky_relu', same_padding=True, use_bn=self.use_bn)
        self.up_path_4_1 = ConBnRelDp(128, 64, 2, stride=2, activate_unit='leaky_relu', same_padding=False, use_bn=self.use_bn,
                                       reverse=True)
        self.up_path_4_2 = ConBnRelDp(64 + 64, 32, 3, stride=1, activate_unit='leaky_relu', same_padding=True, use_bn=self.use_bn)
        self.up_path_4_3 = ConBnRelDp(32, 32, 3, stride=1, activate_unit='leaky_relu', same_padding=True, use_bn=self.use_bn)
        self.up_path_2_1 = ConBnRelDp(32, 32, 2, stride=2, activate_unit='leaky_relu', same_padding=False, use_bn=self.use_bn,
                                       reverse=True)
        self.up_path_2_2 = ConBnRelDp(32 + 32, 16, 3, stride=1, activate_unit='None', same_padding=True)
        self.up_path_2_3 = ConBnRelDp(16, 3, 3, stride=1, activate_unit='None', same_padding=True)
