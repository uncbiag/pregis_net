import SimpleITK as sitk
import os
import numpy as np
import scipy.ndimage.morphology as ndimage
import sys
import glob
import pandas as pd
import cv2

import matplotlib.pyplot as plt


from global_variables import *


def dilate_lowPTV(patient):
    patient_folder = os.path.join(image_root_folder, patient)
    resample_folder = os.path.join(patient_folder, 'resampled_images')
    print("Dilate Low PTV Region to ROI")
    # dilate lowPTV 10mm 
    lowPTV_file = os.path.join(resample_folder, '1900-01__Studies_LowPTV_label.nii.gz')
    lowPTV = sitk.ReadImage(lowPTV_file)
    dilate_filter = sitk.BinaryDilateImageFilter()
    dilate_filter.SetKernelRadius([10,10,10])
    roi = dilate_filter.Execute(lowPTV)
    roi_file = os.path.join(resample_folder, '1900-01__Studies_ROI_label.nii.gz')
    sitk.WriteImage(roi, roi_file)

   
def normalize_image(patient, image_case):
    print("Normalizing Image {} {}".format(patient, image_case))
    patient_folder = os.path.join(image_root_folder, patient)
    resample_folder = os.path.join(patient_folder, 'resampled_images')

    image_file = os.path.join(resample_folder, '19{}-01__Studies_image_3D.nii.gz'.format(image_case))
    I_image = sitk.Cast(sitk.ReadImage(image_file), sitk.sitkFloat32)

    intensity_filter = sitk.IntensityWindowingImageFilter()
    intensity_filter.SetOutputMinimum(0.0)
    intensity_filter.SetOutputMaximum(1.0)

    data_csv = pd.read_csv('data.csv', header=None, dtype=str)
    num_of_images = len(data_csv)
    for idx in range(num_of_images):
        c_patient = data_csv.iloc[idx, 0]
        c_case = data_csv.iloc[idx, 1]
        if c_patient == patient and c_case == image_case:
            windowMinimum = data_csv.iloc[idx, 6]
            windowMaximum = data_csv.iloc[idx, 7]

    if 'windowMinimum' not in locals() or pd.isnull(windowMinimum):
        windowMinimum = int(input("Please input the lower intensity window for image case {}: ".format(image_case)))
    else:
        windowMinimum = int(windowMinimum)
    if 'windowMaximum' not in locals() or pd.isnull(windowMaximum):
        windowMaximum = int(input("Please input the upper intensity window for image case {}: ".format(image_case)))
    else:
        windowMaximum = int(windowMaximum)


    intensity_filter.SetWindowMinimum(windowMinimum)
    intensity_filter.SetWindowMaximum(windowMaximum)

    I_image = intensity_filter.Execute(I_image)
    I_mask = sitk.Cast(sitk.BinaryThreshold(I_image, lowerThreshold=0.01, upperThreshold=1, insideValue=1, outsideValue=0), sitk.sitkUInt8)

    fillhole_filter = sitk.BinaryFillholeImageFilter()
    I_mask = fillhole_filter.Execute(I_mask)

    image_norm_file = os.path.join(resample_folder, '19{}-01__Studies_image_normalized.nii.gz'.format(image_case))
    sitk.WriteImage(I_image, image_norm_file)
    mask_file = os.path.join(resample_folder, '19{}-01__Studies_image_mask.nii.gz'.format(image_case))
    sitk.WriteImage(I_mask, mask_file)


    

def get_contour_from_label(patient, image_case, label_name):
    resampled_folder = os.path.join(image_root_folder, patient, 'resampled_images')
    label_file = os.path.join(resampled_folder, '19{}-01__Studies_{}_label.nii.gz'.format(image_case, label_name))
    I_label = sitk.Cast(sitk.ReadImage(label_file), sitk.sitkUInt8)
    I_label_arr = sitk.GetArrayFromImage(I_label)
    z, x, y = I_label_arr.shape
    I_contour_arr = np.zeros_like(I_label_arr)

    for i in range(z):
        label_slice = I_label_arr[i, :, :]
        contour_slice = np.zeros_like(label_slice, np.uint8)
        if np.sum(label_slice) > 0:
            #plt.imshow(label_slice)
            #plt.show()
            _, contours, _ = cv2.findContours(label_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.polylines(contour_slice, contours, isClosed=True, color=1)
            #plt.imshow(contour_slice)
            #plt.show()
            I_contour_arr[i] = contour_slice

    I_contour = sitk.GetImageFromArray(I_contour_arr)
    I_contour.CopyInformation(I_label)
    contour_file = os.path.join(resampled_folder, '19{}-01__Studies_{}_contour.nii.gz'.format(image_case, label_name))
    print("Writing 19{}-01__Studies_{}_contour.nii.gz".format(image_case, label_name))
    sitk.WriteImage(I_contour, contour_file)

    return I_contour


def get_weighted_mask(patient):
    patient_folder = os.path.join(image_root_folder, patient)
    resampled_folder = os.path.join(patient_folder, 'resampled_images')

    lowPTV_file = os.path.join(resampled_folder, '1900-01__Studies_LowPTV_label.nii.gz')
    highPTV_file = os.path.join(resampled_folder, '1900-01__Studies_HighPTV_label.nii.gz')
    roi_file = os.path.join(resampled_folder, '1900-01__Studies_ROI_label.nii.gz')

    I_lowPTV = sitk.ReadImage(lowPTV_file)
    I_highPTV = sitk.ReadImage(highPTV_file)
    I_roi = sitk.ReadImage(roi_file)

    I_lowPTV_arr = sitk.GetArrayFromImage(I_lowPTV)
    I_highPTV_arr = sitk.GetArrayFromImage(I_highPTV)
    I_roi_arr = sitk.GetArrayFromImage(I_roi)

    I_arr = np.zeros_like(I_lowPTV_arr, dtype=np.float32)
    lowPTV_idx = np.where(I_lowPTV_arr == 1)
    highPTV_idx = np.where(I_highPTV_arr == 1)
    roi_idx = np.where(I_roi_arr == 1)
    I_arr[roi_idx] = 1.0
    I_arr[lowPTV_idx] = 1.0
    I_arr[highPTV_idx] = 1.0

    I_weighted_mask = sitk.GetImageFromArray(I_arr)
    I_weighted_mask.CopyInformation(I_lowPTV)
    sitk.WriteImage(I_weighted_mask, os.path.join(resampled_folder, '1900-01__Studies_weighted_mask_before_smooth.nii.gz'))

    smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    smooth_filter.SetSigma([5,5,5])
    I_weighted_mask = smooth_filter.Execute(I_weighted_mask)

    weighted_mask_file = os.path.join(resampled_folder, "1900-01__Studies_weighted_mask.nii.gz")
    sitk.WriteImage(I_weighted_mask, weighted_mask_file)
    return


def process_labels_to_contour(patient, image_case):
    resampled_folder = os.path.join(image_root_folder, patient, 'resampled_images')

    I_contours = []
    I_contours.append(sitk.Cast(get_contour_from_label(patient, image_case, 'StomachDuo'), sitk.sitkLabelUInt8))
    I_contours.append(sitk.Cast(get_contour_from_label(patient, image_case, 'SmBowel'), sitk.sitkLabelUInt8))
    if image_case == '00':
        I_contours.append(sitk.Cast(get_contour_from_label(patient, image_case, 'HighPTV'), sitk.sitkLabelUInt8))
        I_contours.append(sitk.Cast(get_contour_from_label(patient, image_case, 'LowPTV'), sitk.sitkLabelUInt8))
        I_contours.append(sitk.Cast(get_contour_from_label(patient, image_case, 'ROI'), sitk.sitkLabelUInt8))

    merge_filter = sitk.MergeLabelMapFilter()
    merge_filter.SetMethod(Method=0)

    I_contour_all = sitk.Cast(merge_filter.Execute(I_contours), sitk.sitkUInt8)
    sitk.WriteImage(I_contour_all, os.path.join(resampled_folder, '19{}-01__Studies_all_contours.nii.gz'.format(image_case)))
    return


def crop_image(patient, image_case):
    if image_case == '00':
        return
    patient_folder = os.path.join(image_root_folder, patient)
    cropped_folder = os.path.join(patient_folder, "19{}-01_cropped_images".format(image_case))
    os.system('mkdir -p {}'.format(cropped_folder))
    resampled_folder = os.path.join(patient_folder, 'resampled_images')

    ct_image = sitk.ReadImage(os.path.join(resampled_folder, '1900-01__Studies_image_normalized.nii.gz'))
    cbct_mask = os.path.join(resampled_folder, '19{}-01__Studies_image_mask.nii.gz'.format(image_case))
    mask = sitk.ReadImage(cbct_mask) 
    image_size = np.array(mask.GetSize())
    print(image_size)
    box, crop_range = calculate_crop_range(mask)
    extra_pad = (np.array([480, 448, 224], dtype=np.int64) - np.array(crop_range))//2

    left_extra = (np.array(crop_range) - np.array(box[3:]))//2
    lower_bound = box[0:3] - left_extra
    cropped_lower_bound = np.maximum(lower_bound, 0)
    padded_lower_bound = np.maximum(-lower_bound, 0)
    padded_lower_bound = padded_lower_bound + extra_pad

    upper_bound = image_size-crop_range-lower_bound
    cropped_upper_bound = np.maximum(upper_bound, 0)
    padded_upper_bound = np.maximum(-upper_bound, 0)
    padded_upper_bound = padded_upper_bound + extra_pad

    print(cropped_lower_bound)
    print(padded_lower_bound)
    print(cropped_upper_bound)
    print(padded_upper_bound)
    
    ct_images = glob.glob(os.path.join(resampled_folder, "1900-01*.nii.gz"))
    cbct_images = glob.glob(os.path.join(resampled_folder, "19{}-01*.nii.gz".format(image_case)))
    images = ct_images + cbct_images 

    print(images)
    for img_name in images:
        crop_filter = sitk.CropImageFilter()
        crop_filter.SetUpperBoundaryCropSize(cropped_upper_bound.tolist())
        crop_filter.SetLowerBoundaryCropSize(cropped_lower_bound.tolist())
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound(padded_lower_bound.tolist())
        pad_filter.SetPadUpperBound(padded_upper_bound.tolist())  
 
        processed_image = sitk.ReadImage(img_name)
        name = img_name.split('/')[-1]
        processed_image = crop_filter.Execute(processed_image)
        processed_image = pad_filter.Execute(processed_image)
        new_name = os.path.join(cropped_folder, name)
        processed_image.SetOrigin(ct_image.GetOrigin())
        processed_image.SetDirection(ct_image.GetDirection())
        processed_image.SetSpacing(ct_image.GetSpacing())
        
        print("Writing", new_name)
        sitk.WriteImage(processed_image, new_name)


    print("Cropping CT image based on CBCT Image mask")
    ct_file = os.path.join(cropped_folder, '1900-01__Studies_image_normalized.nii.gz')
    I_ct_image = sitk.ReadImage(ct_file)
    I_ct_array = sitk.GetArrayFromImage(I_ct_image)
    mask_file = os.path.join(cropped_folder, '19{}-01__Studies_image_mask.nii.gz'.format(image_case))
    I_mask = sitk.ReadImage(mask_file)
    mask_arr = sitk.GetArrayFromImage(I_mask)
    mask_idx = np.where(mask_arr == 0)
    I_ct_array[mask_idx] = 0
    I_ct_cropped_image = sitk.GetImageFromArray(I_ct_array)
    I_ct_cropped_image.CopyInformation(I_ct_image)
    ct_cropped_file = os.path.join(cropped_folder, '1900-01__Studies_image_cropped_{}.nii.gz'.format(image_case))
    sitk.WriteImage(I_ct_cropped_image, ct_cropped_file)


def calculate_crop_range(mask):
    shape_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_filter.Execute(mask)
    box = np.array(shape_filter.GetBoundingBox(1), ndmin=2)
    min_x = box[0,3]
    min_y = box[0,4]
    min_z = box[0,5]
    x = min_x + 4 + 32 - (min_x + 4)%32
    y = min_y + 4 + 32 - (min_y + 4)%32
    z = min_z + 4 + 32 - (min_z + 4)%32
  
    print(box[0])
    print(x,y,z)
    return box[0], [x,y,z]


if __name__ == '__main__':
    patient = sys.argv[1]    
    case = sys.argv[2]
    crop_image(patient, case)
    #process_images(patient)
    #normalize_image(patient, case)
    #process_labels_to_contour(patient, case)
    #get_weighted_mask(patient)

