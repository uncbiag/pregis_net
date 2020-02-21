import os
import sys
import glob
import numpy as np
from sklearn.model_selection import KFold
from operator import itemgetter
import socket

def generate_img_list(patients, indices, mode='train'):
    patients = itemgetter(*indices)(patients)
    img_line = ""
    for patient in patients:
        ct_lists = sorted(glob.glob(os.path.join(patient, 'Cropped', 'planCT*')))
        cb_lists = sorted(glob.glob(os.path.join(patient, 'Cropped', 'CBCT*')))
        roi_label = os.path.join(patient, 'Cropped', 'planCT_OG', 'ROI_10.nii.gz')
        roi2_label = os.path.join(patient, 'Cropped', 'planCT_OG', 'ROI_20.nii.gz')
        assert os.path.isfile(roi_label) and os.path.isfile(roi2_label)
        for ct_list in ct_lists:
            ct_image = os.path.join(ct_list, 'image_normalized.nii.gz')
            ct_sblabel = os.path.join(ct_list, 'SmBowel_label.nii.gz')
            ct_sdlabel = os.path.join(ct_list, 'StomachDuo_label.nii.gz')
            assert os.path.isfile(ct_image) and os.path.isfile(ct_sblabel) and os.path.isfile(ct_sdlabel)
            for cb_list in cb_lists:
                cb_image = os.path.join(cb_list, 'image_normalized.nii.gz')
                if mode == 'test':
                    if not 'OG' in ct_image or not 'OG' in cb_image:
                        continue
                cb_sblabel = os.path.join(cb_list, 'SmBowel_label.nii.gz')
                cb_sdlabel = os.path.join(cb_list, 'StomachDuo_label.nii.gz')
                assert os.path.isfile(cb_image) and os.path.isfile(cb_sblabel) and os.path.isfile(cb_sdlabel)
                img_line += "{} {} {} {} {} {} {} {}\n".format(ct_image, cb_image, roi_label,
                                                      ct_sblabel, ct_sdlabel, cb_sblabel, cb_sdlabel, roi2_label)
    return img_line


if __name__ == '__main__':
    root_folder = "/playpen1/xhs400/Research/data/r21/data/ct-cbct/images"
    patients = sorted(glob.glob(os.path.join(root_folder, '18227??')))

    kf = KFold(n_splits=5, shuffle=True)
    fold = 1

    for train_index, test_index in kf.split(patients):
        print(train_index, test_index)
        train_line = generate_img_list(patients, train_index, 'train')
        train_file = "train_{}.txt".format(fold)
        train_f = open(train_file, 'w')
        train_f.write(train_line)
        train_f.close()

        test_line = generate_img_list(patients, test_index, 'test')
        test_file = "test_{}.txt".format(fold)
        test_f = open(test_file, 'w')
        test_f.write(test_line)
        test_f.close()
        fold += 1


