import SimpleITK as sitk
import numpy as np
import re
import os
import sys


def post_process_label(patient, image_case, cases=None):
#    root_folder = '/playpen/xhs400/Research/PycharmProjects/r21/data/ct-cbct/images/' + patient
    pattern = re.compile("^[0-9]{2}$")
    if pattern.match(image_case):
        image_folder = os.path.join(root_folder, '19'+image_case+'-01')
        image_file = os.path.join(image_folder, '19'+image_case+'-01__Studies_image_3D.nii.gz')
        sb_file = os.path.join(image_folder,'19'+image_case+'-01__Studies_SmBowel_mask.nii.gz')
        sd_file = os.path.join(image_folder,'19'+image_case+'-01__Studies_StomachDuo_mask.nii.gz')
        img = sitk.ReadImage(image_file)
        sb_img = sitk.ReadImage(sb_file)
        sd_img = sitk.ReadImage(sd_file)
        img_arr = sitk.GetArrayFromImage(img)
        sb_arr = sitk.GetArrayFromImage(sb_img)
        sd_arr = sitk.GetArrayFromImage(sd_img)
        assert(img_arr.shape == sb_arr.shape and img_arr.shape == sd_arr.shape)
        new_arr = np.zeros_like(img_arr)
        sb_lbl = np.where(sb_arr == 1)
        sd_lbl = np.where(sd_arr == 1)
        new_arr[sb_lbl] = 1
        new_arr[sd_lbl] = 2
        new_img = sitk.GetImageFromArray(new_arr)
        new_img.CopyInformation(img)
        new_file = os.path.join(image_folder,'19'+image_case+'-01__Studies_TwoLabels.nii.gz')
        print(new_file)
        sitk.WriteImage(sitk.Cast(new_img, sitk.sitkUInt8), new_file)
    elif image_case == 'reg_res':
        image_folder = os.path.join(root_folder, image_case, '3D', 'mermaid')
        assert(cases is not None)
        for case in cases:
            if case == '00':
                continue
            sb_file = os.path.join(image_folder, '00-'+case+'-warped-SmBowel-mask.nii.gz')
            sd_file = os.path.join(image_folder, '00-'+case+'-warped-StomachDuo-mask.nii.gz')
            sb_img = sitk.ReadImage(sb_file)
            sd_img = sitk.ReadImage(sd_file)
            sb_arr = sitk.GetArrayFromImage(sb_img)
            sd_arr = sitk.GetArrayFromImage(sd_img)
            assert(sb_arr.shape == sd_arr.shape)
            new_arr = np.zeros_like(sb_arr)
            sb_lbl = np.where(sb_arr > 0)
            sd_lbl = np.where(sd_arr > 0)
            new_arr[sb_lbl] = 1
            new_arr[sd_lbl] = 2
            new_img = sitk.GetImageFromArray(new_arr)
            new_img.CopyInformation(sb_img)
            new_file = os.path.join(image_folder,'00-'+case+'-warped-TwoLabels.nii.gz')
            print(new_file)
            sitk.WriteImage(sitk.Cast(new_img, sitk.sitkUInt8), new_file)
 

    return

def get_ct_case(patient):
    #root_folder = '/playpen/xhs400/Research/PycharmProjects/r21/data/ct-cbct/images/'+patient
    cases = []
    for folder in os.listdir(root_folder):
        if 'reg_res' in folder or ".itksnap" in folder or "DS_Store" in folder:
            continue
        else:
            cases.append(folder[2:4])
    return cases


def main(argv):
    patient = sys.argv[1]
    cases = get_ct_case(patient)
    print(cases)
    for case in cases:
        post_process_label(patient, case)
    #post_process_label(patient, 'reg_res', cases)




if __name__ == '__main__':
    main(sys.argv[1:])

