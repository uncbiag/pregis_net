from __future__ import print_function

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import registration_method
import SimpleITK as sitk
import datetime
import time


mermaid_path = "/playpen/xhs400/Research/FPIR/mermaid"
sys.path.append(mermaid_path)
sys.path.append(os.path.join(mermaid_path, 'pyreg'))
sys.path.append(os.path.join(mermaid_path, 'pyreg/libraries'))

import pyreg.fileio as py_fio
import pyreg.utils as py_utils
from pyreg.data_wrapper import AdaptVal


def run_mermaid(time=None):

    # setup all paths : moving_img, target_img, warped_image, map, momentum
    root_folder = '/playpen/xhs400/Research/PycharmProjects/pregis_net/data_2d_syn'
    if time is not None:
        result_folder = os.path.join(root_folder, 'mermaid_' + time)
    else:
        result_folder = os.path.join(root_folder, 'mermaid')

    os.system('mkdir -p ' + result_folder)
    tumor_folder = os.path.join(result_folder, 'tumor_test')
    no_tumor_folder = os.path.join(result_folder, 'no_tumor_test')
    mask_folder = os.path.join(result_folder, 'mask_test')
    param_folder = os.path.join(result_folder, 'param_folder')
    os.system('mkdir {} {} {} {}'.format(tumor_folder, no_tumor_folder, mask_folder, param_folder))

    im_io = py_fio.ImageIO()
    atlas_image_path = os.path.join(root_folder, 'atlas_2D.nii.gz')
    atlas_image, atlas_hdrc, atlas_spacing, _ = im_io.read_batch_to_nc_format(atlas_image_path, silent_mode=True)

    _id = py_utils.identity_map_multiN(np.array(atlas_image.shape), atlas_spacing)
    identity_map = AdaptVal(torch.from_numpy(_id))
    moving_image_paths = []
    target_image_paths = []
    warped_image_paths = []
    map_paths = []
    momentum_paths = []
    for i in range(100):
        moving_image_paths.append(os.path.join(root_folder, 'no_tumor_test', 'pseudo_no_tumor_test_affined_' + str(i+1) + '.nii.gz'))
        target_image_paths.append(atlas_image_path)
        warped_image_paths.append(os.path.join(no_tumor_folder, 'pseudo_no_tumor_test_mermaid_warped_' + str(i+1) + '.nii.gz'))
        map_paths.append(os.path.join(param_folder, 'pseudo_no_tumor_test_mermaid_map_' + str(i+1) + '.nii.gz'))
        momentum_paths.append(os.path.join(param_folder, 'pseudo_no_tumor_test_momentum_' + str(i+1) + '.npy'))

    moving_images, _, _, _ = im_io.read_batch_to_nc_format(moving_image_paths, silent_mode=True)
    target_images, target_hdrc, target_spacing, _ = im_io.read_batch_to_nc_format(target_image_paths, silent_mode=True)


    warped_images, deformation_maps, momentums = registration_method.image_pair_registration(
            moving_images_w_masks=moving_images,
            target_images_w_masks=target_images,
            target_image_spacing=target_spacing,
            map_resolution=0.5,
            result_folder=result_folder)

    print(deformation_maps.shape)
    print(identity_map.shape)
    for i in range(100):
        im_io.write(warped_image_paths[i], torch.squeeze(warped_images[i,...]), hdr=target_hdrc)

        map_io = py_fio.MapIO()
        displacement_map = deformation_maps[i,...] - identity_map[0,...]
        map_io.write(filename=map_paths[i], data=torch.squeeze(displacement_map), hdr=target_hdrc)
        np.save(momentum_paths[i], momentums[i,...])

        tumor_image_path = os.path.join(root_folder, 'tumor_test', 'pseudo_tumor_test_affined_' + str(i+1) + '.nii.gz')
        mask_image_path = os.path.join(root_folder, 'mask_test', 'pseudo_mask_test_affined_' + str(i+1) + '.nii.gz')
        tumor_image, _, _, _ = im_io.read_batch_to_nc_format(tumor_image_path, silent_mode=True)
        mask_image, _, _, _ = im_io.read_batch_to_nc_format(mask_image_path, silent_mode=True)

        warped_tumor, _= registration_method.evaluate_momentum(torch.from_numpy(tumor_image),
                                                               torch.from_numpy(target_images[[i],...]),
                                                               target_spacing,
                                                               torch.from_numpy(momentums[i,...]), islabel=False)
        warped_tumor_path = os.path.join(tumor_folder, 'pseudo_tumor_test_mermaid_warped_' + str(i+1) + '.nii.gz')
        im_io.write(warped_tumor_path, torch.squeeze(warped_tumor), hdr=target_hdrc)

        warped_mask, _ = registration_method.evaluate_momentum(torch.from_numpy(mask_image),
                                                            torch.from_numpy(target_images[[i],...]),
                                                            target_spacing,
                                                            torch.from_numpy(momentums[i,...]), islabel=True)
        warped_mask_path = os.path.join(mask_folder, 'pseudo_mask_test_mermaid_warped_' + str(i+1) + '.nii.gz')
        im_io.write(warped_mask_path, torch.squeeze(warped_mask), hdr=target_hdrc)


    return


if __name__ == '__main__':
    now = datetime.datetime.now()
    my_time = "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)

    run_mermaid(time=my_time)
