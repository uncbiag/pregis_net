import SimpleITK as sitk
import os
import sys
import numpy as np
import scipy.ndimage.morphology as ndimage


def generate_3D_mask_for_CBCT(patient, image_case):
#    root_folder = '/playpen/xhs400/Research/PycharmProjects/r21/data/ct-cbct/images/'+patient
    image_name = os.path.join(root_folder, '19'+image_case+'-01', '19'+image_case+'-01'+ '__Studies_image_3D.nii.gz')
    I = sitk.ReadImage(image_name)
    max_filter = sitk.MinimumMaximumImageFilter()
    max_filter.Execute(I)
    max_val = max_filter.GetMaximum()
    I_mask = sitk.BinaryThreshold(I, lowerThreshold=0.05, upperThreshold=max_val, insideValue=1, outsideValue=0)
    mask_name = os.path.join(root_folder, '19'+image_case+'-01', '19'+image_case+'-01'+ '__Studies_mask_3D.nii.gz')
    sitk.WriteImage(sitk.Cast(I_mask, sitk.sitkUInt8), mask_name)

    #process_mask(patient, image_case)


def process_mask(patient, image_case):
#    root_folder = '/playpen/xhs400/Research/PycharmProjects/r21/data/ct-cbct/images/'+patient
    mask_name = os.path.join(root_folder, '19'+image_case+'-01', '19'+image_case+'-01'+ '__Studies_mask_3D.nii.gz')
    mask_name_dilated = os.path.join(root_folder, '19'+image_case+'-01', '19'+image_case+'-01'+ '__Studies_mask_dilated_3D.nii.gz')
    I = sitk.ReadImage(mask_name)
    I_arr = sitk.GetArrayFromImage(I)
    num_slices = I_arr.shape[0]
    coronal_sz = I_arr.shape[1]
    struct = ndimage.generate_binary_structure(2,1)
    struct1 = ndimage.iterate_structure(struct, 5)
    for _slice in range(num_slices):
        slice_2D = I_arr[_slice, ...]
        slice_c_first_half = slice_2D[slice(0, int(coronal_sz*0.6)), :]
        slice_c_first_half_tmp = np.copy(slice_c_first_half)
        #slice_c_first_half_tmp[0:int(coronal_sz*0.35), :] = 1
        slice_c_first_half_filled = ndimage.binary_fill_holes(slice_c_first_half_tmp).astype(np.int)
        holes = np.where((slice_c_first_half_filled == 1) & (slice_c_first_half == 0))
        if np.max(slice_2D) > 0:
            processed = ndimage.binary_erosion(ndimage.binary_fill_holes(ndimage.binary_dilation(slice_2D, struct1)), struct1)
            processed[holes] = 0
            I_arr[_slice,...] = processed 
               
    I_new = sitk.GetImageFromArray(I_arr)
    I_new.CopyInformation(I)
    sitk.WriteImage(sitk.Cast(I_new, sitk.sitkUInt8), mask_name_dilated)


if __name__ == '__main__':
    patient = sys.argv[1]
    case = sys.argv[2]
    generate_3D_mask_for_CBCT(patient, case)

