import os
import sys
import glob
import numpy as np
from sklearn.model_selection import KFold
from operator import itemgetter


def generate_img_list(patients, indices):
    patients = itemgetter(*indices)(patients)
    img_line = ""
    for patient in patients:
        ct_lists = sorted(glob.glob(os.path.join(patient, 'planCT*')))
        roi_label = os.path.join(patient, 'planCT_OG', 'roi_labels.nii.gz')
        assert os.path.isfile(roi_label)
        for ct_list_1 in ct_lists:
            ct_image_1 = os.path.join(ct_list_1, 'image_normalized.nii.gz')
            ct_sblabel_1 = os.path.join(ct_list_1, 'sb_labels.nii.gz')
            ct_sdlabel_1 = os.path.join(ct_list_1, 'sd_labels.nii.gz')
            assert os.path.isfile(ct_image_1) and os.path.isfile(ct_sblabel_1) and os.path.isfile(ct_sdlabel_1)
            for ct_list_2 in ct_lists:
                ct_image_2 = os.path.join(ct_list_2, 'image_normalized.nii.gz')
                ct_sblabel_2 = os.path.join(ct_list_2, 'sb_labels.nii.gz')
                ct_sdlabel_2 = os.path.join(ct_list_2, 'sd_labels.nii.gz')
                assert os.path.isfile(ct_image_2) and os.path.isfile(ct_sblabel_2) and os.path.isfile(ct_sdlabel_2)
                img_line += "{} {} {} {} {} {} {}\n".format(ct_image_1, ct_image_2, roi_label,
                                                      ct_sblabel_1, ct_sdlabel_1, ct_sblabel_2, ct_sdlabel_2)
    return img_line


if __name__ == '__main__':
    root_folder = "/playpen1/xhs400/Research/data/r21/data/ct-cbct/images/all"
    patients = sorted(glob.glob(os.path.join(root_folder, '18227*')))
    print(patients)

    kf = KFold(n_splits=5, shuffle=True)
    fold = 1
    for train_index, test_index in kf.split(patients):
        train_line = generate_img_list(patients, train_index)
        train_file = "train_{}.txt".format(fold)
        train_f = open(train_file, 'w')
        train_f.write(train_line)
        train_f.close()

        test_line = generate_img_list(patients, test_index)
        test_file = "test_{}.txt".format(fold)
        test_f = open(test_file, 'w')
        test_f.write(test_line)
        test_f.close()
        fold += 1


