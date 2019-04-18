import os
import sys
import numpy as np
import torch

mermaid_path = "/playpen/xhs400/Research/FPIR/mermaid"
sys.path.append(mermaid_path)
sys.path.append(os.path.join(mermaid_path, 'pyreg'))
sys.path.append(os.path.join(mermaid_path, 'pyreg/libraries'))

import pyreg.fileio as py_fio
import pyreg.image_sampling as py_is
import pyreg.model_factory as py_mf
import pyreg.simple_interface as py_si
import pyreg.utils as py_utils
import pyreg.visualize_registration_results as py_visreg
import pyreg.module_parameters as pars
from pyreg.data_wrapper import AdaptVal


def evaluate_momentum(moving_image, target_image, target_spacing, momentum, islabel=False):
    shared_parameters=dict()
    params=pars.ParameterDict()
    print(sys.path[0])
    registration_param_file=os.path.join(os.path.dirname(__file__), '../settings/mermaid_config.json')
    print(registration_param_file)
    params.load_JSON(registration_param_file)

    momentum_dict=dict()
    #momentum_dict['m'] = torch.from_numpy(momentum)
    momentum_dict['m'] = momentum
    warped_image,map_from_momentum, _ = evaluate_model(
        ISource_in = moving_image,
        ITarget_in = target_image,
        sz = np.shape(target_image),
        spacing=target_spacing,
        individual_parameters=momentum_dict,
        shared_parameters=shared_parameters,
        params=params,
        visualize=False,
        compute_inverse_map=False,
        islabel=islabel
    )

    return warped_image, map_from_momentum



def image_pair_registration(moving_images_w_masks, target_images_w_masks, target_image_spacing, map_resolution, result_folder):
    """
    :param moving_image_paths: path for moving image
    :param target_image_paths: path for target image
    :param result_image_paths: path for result warped image, default None
    :param result_deformation_paths: path for deformation, default None
    :return: moving_images, target_images, momentum
    """
    si = py_si.RegisterImagePair()
    si.register_images(moving_images_w_masks, target_images_w_masks, target_image_spacing,
                       visualize_step=None,
                       model_name='svf_vector_momentum_map',
                       use_multi_scale=False,
                       use_batch_optimization=False,
                       json_config_out_filename=os.path.join(result_folder, 'mermaid_config.json'),
                       params=os.path.join(os.path.dirname(__file__), '../settings/mermaid_config.json'),
                       map_low_res_factor=map_resolution)

    warped_images_w_masks = si.get_warped_image()
    deformation_map = si.get_map()
    model_pars = si.get_model_parameters()
    momentum = model_pars['m'].cpu().data.numpy()
    return warped_images_w_masks, deformation_map, momentum


def _compute_low_res_image(I,spacing,low_res_size):
    sampler = py_is.ResampleImage()
    low_res_image, _ = sampler.downsample_image_to_size(I, spacing, low_res_size[2::],1)
    return low_res_image


def _get_low_res_size_from_size(sz, factor):
    """
    Returns the corresponding low-res size from a (high-res) sz
    :param sz: size (high-res)
    :param factor: low-res factor (needs to be <1)
    :return: low res size
    """
    if (factor is None) or (factor>=1):
        print('WARNING: Could not compute low_res_size as factor was ' + str( factor ))
        return sz
    else:
        lowResSize = np.array(sz)
        lowResSize[2::] = (np.ceil((np.array(sz[2::]) * factor))).astype('int16')

        #if lowResSize[-1]%2!=0:
        #    lowResSize[-1]-=1
        #    print('\n\nWARNING: forcing last dimension to be even: fix properly in the Fourier transform later!\n\n')

        return lowResSize


def _get_low_res_spacing_from_spacing(spacing, sz, lowResSize):
    """
    Computes spacing for the low-res parameterization from image spacing
    :param spacing: image spacing
    :param sz: size of image
    :param lowResSize: size of low re parameterization
    :return: returns spacing of low res parameterization
    """
    #todo: check that this is the correct way of doing it
    return spacing * (np.array(sz[2::])-1) / (np.array(lowResSize[2::])-1)


def evaluate_model(ISource_in,ITarget_in,sz,spacing,individual_parameters,shared_parameters,params,visualize=True,
                   compute_inverse_map=False, islabel=False):
    """
    evaluates model for given source and target image, size and spacing
    model specified by individual and shared parameters, as well as params from loaded jsonfile
    :param ISource_in: source image as torch variable
    :param ITarget_in: target image as torch variable
    :param sz: size of images
    :param spacing: spacing of images
    :param individual_parameters: dictionary containing the momentum
    :param shared_parameters: empty dictionary
    :param params: model parameter from loaded jsonfile
    :param visualize: if True - plots IS,IT,IW,chessboard,grid,momentum
    :param compute_inverse_map: if true - gives out inverse deformation map [inverse map = None if False]
    :return: returns IWarped, map, inverse map as torch variables
    """

    ISource = AdaptVal(ISource_in)
    ITarget = AdaptVal(ITarget_in)

    model_name = params['model']['registration_model']['type']
    use_map = params['model']['deformation']['use_map']
    map_low_res_factor = params['model']['deformation'][('map_low_res_factor', None, 'low_res_factor')]
    compute_similarity_measure_at_low_res = params['model']['deformation'][
        ('compute_similarity_measure_at_low_res', False, 'to compute Sim at lower resolution')]
    print(map_low_res_factor)
    lowResSize = None
    lowResSpacing = None
    ##
    if map_low_res_factor == 1.0:
        map_low_res_factor = None
    ##
    if map_low_res_factor is not None:
        lowResSize = _get_low_res_size_from_size(sz, map_low_res_factor)
        lowResSpacing = _get_low_res_spacing_from_spacing(spacing, sz, lowResSize)

        lowResISource = _compute_low_res_image(ISource, spacing, lowResSize)
        # todo: can be removed to save memory; is more experimental at this point
        lowResITarget = _compute_low_res_image(ITarget, spacing, lowResSize)

    if map_low_res_factor is not None:
        # computes model at a lower resolution than the image similarity
        if compute_similarity_measure_at_low_res:
            mf = py_mf.ModelFactory(lowResSize, lowResSpacing, lowResSize, lowResSpacing)
        else:
            mf = py_mf.ModelFactory(sz, spacing, lowResSize, lowResSpacing)
    else:
        # computes model and similarity at the same resolution
        mf = py_mf.ModelFactory(sz, spacing, sz, spacing)

    model, criterion = mf.create_registration_model(model_name, params['model'], compute_inverse_map)
    # set it to evaluation mode
    model.eval()

    if use_map:
        # create the identity map [-1,1]^d, since we will use a map-based implementation
        _id = py_utils.identity_map_multiN(sz, spacing)
        identityMap = AdaptVal(torch.from_numpy(_id))
        if map_low_res_factor is not None:
            # create a lower resolution map for the computations
            lowres_id = py_utils.identity_map_multiN(lowResSize, lowResSpacing)
            lowResIdentityMap = AdaptVal(torch.from_numpy(lowres_id))

    if False:
        model = model.cuda()

    dictionary_to_pass_to_integrator = dict()

    if map_low_res_factor is not None:
        dictionary_to_pass_to_integrator['I0'] = lowResISource
        dictionary_to_pass_to_integrator['I1'] = lowResITarget
    else:
        dictionary_to_pass_to_integrator['I0'] = ISource
        dictionary_to_pass_to_integrator['I1'] = ITarget

    model.set_dictionary_to_pass_to_integrator(dictionary_to_pass_to_integrator)

    model.set_shared_registration_parameters(shared_parameters)
    ##model_pars = individual_parameters_to_model_parameters(individual_parameters)
    model_pars = individual_parameters
    model.set_individual_registration_parameters(model_pars)

    # now let's run the model
    rec_IWarped = None
    rec_phiWarped = None
    rec_phiWarped_inverse = None

    if islabel:
        spline_order = 0
    else:
        spline_order = 1
    if use_map:
        if map_low_res_factor is not None:
            if compute_similarity_measure_at_low_res:
                maps = model(lowResIdentityMap, lowResISource)
                if compute_inverse_map:
                    rec_phiWarped = maps[0]
                    rec_phiWarped_inverse=maps[1]
                else:
                    rec_phiWarped = maps
            else:
                maps = model(lowResIdentityMap, lowResISource)
                if compute_inverse_map:
                    rec_tmp = maps[0]
                    rec_tmp_inverse=maps[1]
                else:
                    rec_tmp = maps
                # now up-sample to correct resolution
                desiredSz = identityMap.size()[2::]
                sampler = py_is.ResampleImage()
                rec_phiWarped, _ = sampler.upsample_image_to_size(rec_tmp, spacing, desiredSz,spline_order=spline_order)
                if compute_inverse_map:
                    rec_phiWarped_inverse, _ = sampler.upsample_image_to_size(rec_tmp_inverse, spacing, desiredSz, spline_order=spline_order)
        else:
            maps = model(identityMap, ISource)
            if compute_inverse_map:
                rec_phiWarped = maps[0]
                rec_phiWarped_inverse = maps[1]
            else:
                rec_phiWarped = maps

    else:
        rec_IWarped = model(ISource)

    if use_map:
        rec_IWarped = py_utils.compute_warped_image_multiNC(ISource, rec_phiWarped, spacing, spline_order=spline_order)

    if use_map and map_low_res_factor is not None:
        vizImage, vizName = model.get_parameter_image_and_name_to_visualize(lowResISource)
    else:
        vizImage, vizName = model.get_parameter_image_and_name_to_visualize(ISource)

    if use_map:
        phi_or_warped_image = rec_phiWarped
    else:
        phi_or_warped_image = rec_IWarped

    visual_param = {}
    visual_param['visualize'] = visualize
    visual_param['save_fig'] = False

    if use_map:
        if compute_similarity_measure_at_low_res:
            I1Warped = py_utils.compute_warped_image_multiNC(lowResISource, phi_or_warped_image, lowResSpacing,spline_order=spline_order)
            py_visreg.show_current_images(iter, lowResISource, lowResITarget, I1Warped, vizImage, vizName,
                                       phi_or_warped_image, visual_param)
        else:
            I1Warped = py_utils.compute_warped_image_multiNC(ISource, phi_or_warped_image, spacing,1)
            #py_visreg.show_current_images(iter=iter, iS=ISource, iT=ITarget, iW=I1Warped, vizImages=vizImage, vizName=vizName,
                                          #phiWarped=phi_or_warped_image, visual_param=visual_param)
    else:
        py_visreg.show_current_images(iter, ISource, ITarget, phi_or_warped_image, vizImage, vizName, None, visual_param)

    return rec_IWarped, rec_phiWarped, rec_phiWarped_inverse

