from __future__ import print_function

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import SimpleITK as sitk
import datetime
import time
import glob


mermaid_path = '/playpen/xhs400/Research/PycharmProjects/r21_net/mermaid'
sys.path.append(mermaid_path)
sys.path.append(os.path.join(mermaid_path, 'pyreg'))
sys.path.append(os.path.join(mermaid_path, 'pyreg/libraries'))
import pyreg.fileio as py_fio
import pyreg.utils as py_utils
import pyreg.simple_interface as py_si
from pyreg.data_wrapper import AdaptVal
import argparse


def image_pair_registration(moving_images, target_images, target_image_spacing, map_resolution, result_folder, registration_param_file, model_name=None, compute_inverse_map=False):
    """
    :param moving_image_paths: path for moving image
    :param target_image_paths: path for target image
    :param result_image_paths: path for result warped image, default None
    :param result_deformation_paths: path for deformation, default None
    :return: moving_images, target_images, momentum
    """
    si = py_si.RegisterImagePair()
    si.register_images(moving_images, target_images, target_image_spacing,
                       visualize_step=None,
                       model_name=model_name,
                       use_multi_scale=False,
                       use_batch_optimization=False,
                       json_config_out_filename=os.path.join(result_folder, 'mermaid_config.json'),
                       params=registration_param_file,
                       map_low_res_factor=map_resolution,
                       compute_inverse_map=compute_inverse_map)

    warped_images = si.get_warped_image()
    model_pars = si.get_model_parameters()
    momentum = model_pars['m'].cpu().data.numpy()
    forward_map = si.get_map()
    if compute_inverse_map:
        inverse_map = si.get_inverse_map()
        return warped_images, momentum, forward_map, inverse_map
    else:
        return warped_images, momenum, forward_map


def mermaid_reg(moving_image_file, target_image_file, warped_image_file, forward_disp_file, moving_label_files=None, warped_label_files=None, mask_file=None, inverse_disp_file=None, momentum_file=None, json_file=None):
    result_folder= os.path.dirname(warped_image_file)
    im_io = py_fio.ImageIO()
    map_io = py_fio.MapIO()

    moving_image, moving_hdrc, moving_spacing, _ = im_io.read_batch_to_nc_format(moving_image_file)
    _id_moving = py_utils.identity_map_multiN(np.array(moving_image.shape), moving_spacing)
    identity_map_moving = AdaptVal(torch.from_numpy(_id_moving))

    target_image, target_hdrc, target_spacing, _ = im_io.read_batch_to_nc_format(target_image_file)
    _id_target = py_utils.identity_map_multiN(np.array(target_image.shape), target_spacing)
    identity_map_target = AdaptVal(torch.from_numpy(_id_target))

    if mask_file is not None:
        mask_image, _, _, _ = im_io.read_batch_to_nc_format(mask_file, silent_mode=True)
        target_image = np.concatenate((target_image, mask_image), axis=1)
        print("Target Shape {}".format(target_image.shape))

    model = json_file.split('mermaid_config_')[1]
    print(model)
    if model == "lddmm.json":
        model_name = "lddmm_shooting_map"
    elif model == "svf.json":
        model_name = "svf_vector_momentum_map"
    else:
        raise ValueError("Error")

    warped_image, momentum, forward_map, inverse_map = image_pair_registration(moving_image, target_image, target_spacing, 0.5, result_folder, json_file, model_name, compute_inverse_map=True)
    im_io.write(warped_image_file, torch.squeeze(warped_image[0,...]), hdr=target_hdrc)

    np.save(momentum_file, momentum)
    forward_disp_map = forward_map[0,...] - identity_map_target[0,...]
    inverse_disp_map = inverse_map[0,...] - identity_map_moving[0,...]
    map_io.write(filename=forward_disp_file, data=torch.squeeze(forward_disp_map), hdr=target_hdrc)
    map_io.write(filename=inverse_disp_file, data=torch.squeeze(inverse_disp_map), hdr=moving_hdrc)

    if moving_label_files is not None:
        moving_labels, _, moving_spacing, _ = im_io.read_batch_to_nc_format(moving_label_files)
        moving_labels = np.swapaxes(moving_labels, 0, 1)
        print(moving_labels.shape)
        warped_labels = py_utils.compute_warped_image_multiNC(torch.from_numpy(moving_labels).cuda(), forward_map, target_spacing, spline_order=0, zero_boundary=True)
        print(warped_labels.shape)
        for i in range(len(warped_label_files)):
            im_io.write(warped_label_files[i], torch.squeeze(warped_labels[:,i, ...]), hdr=target_hdrc)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mermaid")
    parser.add_argument("--moving", action="store", dest="moving_image_file")
    parser.add_argument("--target", action="store", dest="target_image_file")
    parser.add_argument("--warped", action="store", dest="warped_image_file")
    parser.add_argument("--disp", action="store", dest="forward_disp_file")
    parser.add_argument("--inv", action="store", dest="inverse_disp_file")
    parser.add_argument("--momentum", action="store", dest="momentum_file")
    parser.add_argument("--json", action="store", dest="json_file")
    parser.add_argument("--mask", action="store", dest="mask_file")
    parser.add_argument('--labels', nargs='+', dest='label_files', help="labels to warp")
    parser.add_argument('--warped_labels', nargs='+', dest='warped_labels', help="warped label files")

    args = parser.parse_args()
    print(args)
    moving_image_file = args.moving_image_file
    target_image_file = args.target_image_file
    warped_image_file = args.warped_image_file
    forward_disp_file = args.forward_disp_file
    inverse_disp_file = args.inverse_disp_file
    momentum_file = args.momentum_file
    label_files = args.label_files
    warped_labels = args.warped_labels
    if label_files is not None:
        assert(len(label_files) == len(warped_labels))
    json_file = args.json_file
    mask_file = args.mask_file

    mermaid_reg(moving_image_file=moving_image_file,
                target_image_file=target_image_file,
                warped_image_file=warped_image_file,
                forward_disp_file=forward_disp_file,
                inverse_disp_file=inverse_disp_file,
                mask_file=mask_file,
                momentum_file=momentum_file,
                moving_label_files=label_files,
                warped_label_files=warped_labels,
                json_file=json_file)

