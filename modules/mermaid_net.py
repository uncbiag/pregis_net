import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('../')
from utils.registration_method import _get_low_res_size_from_size, _get_low_res_spacing_from_spacing, _compute_low_res_image

import pyreg.module_parameters as pars
import pyreg.model_factory as py_mf
import pyreg.utils as py_utils
import pyreg.image_sampling as py_is
import pyreg.similarity_measure_factory as smf

from modules.u_net import UNet


class MermaidNet(nn.Module):
    def __init__(self, model_config):
        super(MermaidNet, self).__init__()
        self.config = model_config

        bn = self.config['bn']
        dp = self.config['dp']
        dim = self.config['dim']
        self.img_sz = self.config['img_sz']
        self.batch_size = self.img_sz[0]

        self.u_net = UNet(dim=dim, use_bn=bn, use_dp=dp)
        self.mermaid_config_file = model_config['mermaid_config_file']
        self.mermaid_unit = None
        self.spacing = 1./(np.array(self.img_sz[2:])-1)
        self.init_mermaid_env(spacing=self.spacing)
        return


    def init_mermaid_env(self, spacing):
        params = pars.ParameterDict()
        params.load_JSON(self.mermaid_config_file)
        sm_factory = smf.SimilarityMeasureFactory(self.spacing)
        self.sim_criterion = sm_factory.create_similarity_measure(params['model']['registration_model'])
        model_name = params['model']['registration_model']['type']
        self.use_map = params['model']['deformation']['use_map']
        self.map_low_res_factor = params['model']['deformation'][('map_low_res_factor', None, 'low_res_factor')]
        compute_similarity_measure_at_low_res = params['model']['deformation'][('compute_similarity_measure_at_low_res', False, 'to compute Sim at lower resolution')]

        ## Currently Must use map_low_res_factor = 0.5
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


        ## Currently Must use map
        if self.use_map:
            # create the identity map [0,1]^d, since we will use a map-based implementation
            _id = py_utils.identity_map_multiN(self.img_sz, spacing)
            self.identityMap = torch.from_numpy(_id).cuda()
            if self.map_low_res_factor is not None:
                # create a lower resolution map for the computations
                lowres_id = py_utils.identity_map_multiN(self.lowResSize, self.lowResSpacing)
                self.lowResIdentityMap = torch.from_numpy(lowres_id).cuda()
        self.mermaid_unit = model.cuda()
        self.mermaid_criterion = criterion
        self.sampler = py_is.ResampleImage()

        return


    def cal_mermaid_loss(self, moving, target):
        loss_dict = {}

        mermaid_all_loss, mermaid_sim_loss, mermaid_reg_loss = self.mermaid_criterion(phi0=self.identityMap,
                                                                                      phi1=self.phi,
                                                                                      I0_source=moving,
                                                                                      I1_target=target,
                                                                                      lowres_I0=None,
                                                                                      variables_from_forward_model=self.mermaid_unit.get_variables_to_transfer_to_loss_function(),
                                                                                      variables_from_optimizer=None)

        loss_dict['mermaid_all_loss'] = mermaid_all_loss
        loss_dict['mermaid_sim_loss'] = mermaid_sim_loss
        loss_dict['mermaid_reg_loss'] = mermaid_reg_loss
        return loss_dict

    def set_mermaid_params(self, moving, target, momentum):
        self.mermaid_unit.set_dictionary_to_pass_to_integrator({'I0': moving, 'I1': target})
        self.mermaid_criterion.set_dictionary_to_pass_to_smoother({'I0': moving, 'I1': target})
        self.mermaid_unit.m = momentum
        self.mermaid_criterion.m = momentum
        return


    def mermaid_shoot(self, moving, target, momentum):
        self.set_mermaid_params(moving=moving, target=target, momentum=momentum)
        lowResMoving = _compute_low_res_image(target, self.spacing, self.lowResSize)
        lowResPhi = self.mermaid_unit(self.lowResIdentityMap, lowResMoving)
        desiredSz = self.identityMap.size()[2:]
        phi, _ = self.sampler.upsample_image_to_size(lowResPhi, self.spacing, desiredSz, spline_order=1)
        moving_warped = py_utils.compute_warped_image_multiNC(moving, phi, self.spacing, spline_order=1)
        return moving_warped, phi


    def single_forward(self, moving, target):
        momentum = self.u_net(moving, target) # get momentum
        warped, phi = self.mermaid_shoot(moving, target, momentum)
        self.warped = warped
        self.phi = phi
        return


    def forward(self, moving, target):
        self.single_forward(moving, target)
        return