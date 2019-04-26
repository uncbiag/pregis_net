import torch
import torch.nn as nn
import numpy as np
import os
import sys
import glob
sys.path.append('../')
from utils.registration_method import _get_low_res_size_from_size, _get_low_res_spacing_from_spacing, _compute_low_res_image

import pyreg.module_parameters as pars
import pyreg.model_factory as py_mf
import pyreg.utils as py_utils
import pyreg.image_sampling as py_is
import pyreg.fileio as py_fio
import pyreg.similarity_measure_factory as smf

import losses.loss as loss
from modules.vae import VaeNet
from modules.momentum_net import MomentumNet



class PregisNet(nn.Module):
    def __init__(self, model_config):
        super(PregisNet, self).__init__()
        self.config = model_config
        model_name = self.config['name']
        assert model_name == 'pregis_net'  # the only model available
        self.pretrain_epochs = self.config[model_name]['pretrain_epochs']
        self.join_two_networks = self.config[model_name]['join_two_networks']
        n_ft=self.config['n_of_feature']
        bn=self.config['bn']
        dp=self.config['dp']
        dim = self.config['dim']

    
        self.img_sz = self.config['img_sz']
        self.batch_size = self.img_sz[0]
        
        if self.config[model_name]['momentum_net']['name'] == "MomentumNet":
            self.momentum_net = MomentumNet(n_ft=n_ft, dim=dim)
        else:
            raise ValueError("Not supported")
        if self.config[model_name]['recons_net']['name'] == "VaeNet":    
            self.recons_net = VaeNet(dim=dim)
        else:
            raise ValueError("Not supported")

        use_TV_loss = self.config[model_name]['recons_net']['use_TV_loss']
        self.gamma_recons = self.config[model_name]['recons_net']['gamma_recons']
        self.gamma_mermaid = self.config[model_name]['momentum_net']['gamma_mermaid']
         
        self.recons_criterion_L1 = nn.L1Loss(reduction='mean').cuda()
        if use_TV_loss:
            self.recons_criterion_TV = loss.TVLoss().cuda()
        else:
            self.recons_criterion_TV = None 

        self.mermaid_unit = None
        self.use_map = True
        self.map_low_res_factor = 0.5

        self.spacing = 1./(np.array(self.img_sz[2:])-1)
        self.init_mermaid_env(spacing=self.spacing)

        self.rec_phiWarped = None

        self.img_recons = None
        self.momentum = None
        return    

    #######################
    #Initialize Mermaid Unit (Shooting)
    #######################

    def init_mermaid_env(self, spacing):
        params = pars.ParameterDict()
        params.load_JSON(os.path.join(os.path.dirname(__file__), '../settings/mermaid_config.json'))
       
        sm_factory = smf.SimilarityMeasureFactory(self.spacing)
        self.sim_criterion = sm_factory.create_similarity_measure(params['model']['registration_model'])
        model_name = params['model']['registration_model']['type']
        self.use_map = params['model']['deformation']['use_map']
        self.map_low_res_factor = params['model']['deformation'][('map_low_res_factor', None, 'low_res_factor')]
        compute_similarity_measure_at_low_res = params['model']['deformation'][('compute_similarity_measure_at_low_res', False, 'to compute Sim at lower resolution')]
        sim_sigma = params['model']['registration_model']['similarity_measure']['sigma']


        ## Currently Must use map_low_res_factor = 0.5
        if self.map_low_res_factor is not None:
            self.lowResSize = _get_low_res_size_from_size(self.img_sz, self.map_low_res_factor)
            self.lowResSpacing = _get_low_res_spacing_from_spacing(spacing, self.img_sz, self.lowResSize)
            if compute_similarity_measure_at_low_res:
                mf = py_mf.ModelFactory(self.lowResSize, self.lowResSpacing, self.lowResSize, self.lowResSpacing)
            else:
                mf = py_mf.ModelFactory(self.img_sz, self.spacing, self.lowResSize, self.lowResSpacing)

        model, criterion = mf.create_registration_model(model_name, params['model'], compute_inverse_map=True)


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

        return

    #########################
    #  Calcuate Loss functions
    #########################


    def cal_pregis_loss(self, moving, target, cur_epoch, mode):
        mermaid_all_loss, mermaid_sim_loss, mermaid_reg_loss = self.mermaid_criterion(phi0=self.identityMap,phi1=self.phi,I0_source=moving,I1_target=target,lowres_I0=None,variables_from_forward_model=self.mermaid_unit.get_variables_to_transfer_to_loss_function(),variables_from_optimizer=None)

        loss_dict = {}
        loss_dict['mermaid_sim_loss'] = mermaid_sim_loss
        loss_dict['mermaid_reg_loss'] = mermaid_reg_loss

        # factors controlling the balance for mermaid sim loss and reg sim loss
        fine_factor = 1./(np.exp((self.pretrain_epochs-cur_epoch)/self.pretrain_epochs*10)+1)
        pre_factor = 1./(np.exp((cur_epoch-self.pretrain_epochs)/self.pretrain_epochs*10)+1)        
        recons_sim_loss = self.sim_criterion.compute_similarity_multiNC(self.moving_warped_recons, target)
        loss_dict['recons_sim_loss'] = recons_sim_loss 
        sim_loss = pre_factor*mermaid_sim_loss + fine_factor*recons_sim_loss
        loss_dict['sim_loss'] = sim_loss

        if self.join_two_networks:
            # controlling recons loss: do not backpropogate through momentum net
            # using moving_warped_recons_det, which is obtained from detached movining warped
            recons_loss_L1 = self.recons_criterion_L1(self.moving_warped.detach(), self.moving_warped_recons_det)
            loss_dict['recons_loss_L1'] = recons_loss_L1
            if self.recons_criterion_TV is not None:
                recons_loss_TV = self.recons_criterion_TV(self.moving_warped.detach(), self.moving_warped_recons_det)
                loss_dict['recons_loss_TV'] = recons_loss_TV
                recons_loss = recons_loss_L1 + recons_loss_TV
            else:
                recons_loss = recons_loss_L1
            loss_dict['recons_loss'] = recons_loss 
        else:
            recons_loss_L1 = self.recons_criterion_L1(moving, self.moving_warped_recons)
            loss_dict['recons_loss_L1'] = recons_loss_L1
            if self.recons_criterion_TV is not None:
                recons_loss_TV = self.recons_criterion_TV(moving, self.moving_warped_recons)
                loss_dict['recons_loss_TV'] = recons_loss_TV
                recons_loss = recons_loss_L1 + recons_loss_TV
            else:
                recons_loss = recons_loss_L1
            loss_dict['recons_loss'] = recons_loss 
             
        # mu, logvar have been detached
        kld_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
        kld_loss = torch.mean(kld_element).mul_(-0.5)
        loss_dict['kld_loss'] = kld_loss 

        if mode == 'train':
            # training loss, a combination of mermaid sim and recons sim
            all_loss = self.gamma_mermaid*(mermaid_reg_loss + sim_loss)/self.batch_size + self.gamma_recons* (0.001*kld_loss + recons_loss) 
        elif mode == 'validate':
            # validate loss, only considering recons sim loss
            all_loss = self.gamma_mermaid*(mermaid_reg_loss + recons_sim_loss)/self.batch_size + self.gamma_recons* (0.001*kld_loss + recons_loss)
        else:
            raise ValueError("mode not recognized")
        loss_dict['all_loss'] = all_loss

        return loss_dict
    

    #####################################
    #  Forward shooting 
    #####################################

    def set_mermaid_params(self, moving, target, momentum):
        self.mermaid_unit.set_dictionary_to_pass_to_integrator({'I0': moving, 'I1': target})
        self.mermaid_criterion.set_dictionary_to_pass_to_smoother({'I0': moving, 'I1': target})
        self.mermaid_unit.m = momentum
        self.mermaid_criterion.m = momentum
        return 

    def mermaid_shoot(self, moving, target, momentum): # target to recons
        if self.map_low_res_factor is not None:
            self.set_mermaid_params(moving=moving, target=target, momentum=momentum)
            lowResMoving = _compute_low_res_image(target, self.spacing, self.lowResSize)
            # TODO inverse initial map should base on moving image space, not lowResIdentityMap
            phi, phi_inv = self.mermaid_unit(self.lowResIdentityMap, lowResMoving, self.lowResIdentityMap)

            desiredSz = self.identityMap.size()[2:]
            sampler = py_is.ResampleImage()
            rec_phi, _ = sampler.upsample_image_to_size(phi, self.spacing, desiredSz, spline_order=1)
            rec_phi_inv, _ = sampler.upsample_image_to_size(phi_inv, self.spacing, desiredSz, spline_order=1)
            

        if self.use_map:
            moving_warped = py_utils.compute_warped_image_multiNC(moving, rec_phi, self.spacing, spline_order=1)
        return moving_warped, rec_phi, rec_phi_inv

    #####################################
    #  Forward pass
    #####################################

    def single_forward(self, moving,target, current_epoch):
        momentum = self.momentum_net(moving, target)
        moving_warped, phi, phi_inv = self.mermaid_shoot(moving, target, momentum)

        if self.join_two_networks:
            # forward vae, detached from momentum net
            moving_warped_recons_det, mu_det, logvar_det = self.recons_net(moving_warped.detach())
            self.moving_warped_recons_det = moving_warped_recons_det
            self.mu = mu_det
            self.logvar = logvar_det

            # forward vae, attached to momentum net
            moving_warped_recons, _, _ = self.recons_net(moving_warped)

        else: # use moving image
            moving_warped_recons, mu, logvar = self.recons_net(moving)
            self.mu = mu
            self.logvar = logvar
 
        self.moving_warped_recons = moving_warped_recons
        self.phi = phi
        self.phi_inv = phi_inv

        self.momentum = momentum
        self.moving_warped = moving_warped

        return moving_warped.detach(), moving_warped_recons.detach(), phi.detach()



    def forward(self, moving, target, current_epoch=0):
        return self.single_forward(moving, target, current_epoch)

