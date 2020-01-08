import pydicom
from dicom_contour.contour import *
import SimpleITK as sitk
import sys
import os
import os.path
import numpy as np
import pandas as pd

from global_variables import *

def get_contour_file(path):
    """
    Get contour file from a given path by searching for ROIContourSequence 
    inside dicom data structure.
    More information on ROIContourSequence available here:
    http://dicom.nema.org/medical/dicom/2016c/output/chtml/part03/sect_C.8.8.6.html
    
    Inputs:
            path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
    Return:
        contour_file (str): name of the file with the contour
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # get .dcm contour file
    fpaths = [path + f for f in os.listdir(path) if '.dcm' in f]
    n = 0
    contour_file = None
    for fpath in fpaths:
        f = pydicom.read_file(fpath)
        if 'ROIContourSequence' in dir(f):
            contour_file = fpath.split('/')[-1]
            n += 1
    if n > 1: warnings.warn("There are multiple files, returning the last one!")
    return contour_file



def slice_order(path):
    """
    Takes path of directory that has the DICOM images and returns
    a ordered list that has ordered filenames
    Inputs
        path: path that has .dcm images
    Returns
        ordered_slices: ordered tuples of filename and z-position
    """
    # handle `/` missing
    if path[-1] != '/': path += '/'
    slices = []
    for s in os.listdir(path):
        try:
            f = dicom.read_file(path + '/' + s)
            f.pixel_array  # to ensure not to read contour file
            slices.append(f)
        except:
            continue

    slice_dict = {s.SOPInstanceUID: s.ImagePositionPatient[-1] for s in slices}
    ordered_slices = sorted(slice_dict.items(), key=operator.itemgetter(1))
    return ordered_slices



def get_data(path, index):
    """
    Generate image array and contour array
    Inputs:
        path (str): path of the the directory that has DICOM files in it
        contour_dict (dict): dictionary created by get_contour_dict
        index (int): index of the desired ROISequence
    Returns:
        images and contours np.arrays
    """
    images = []
    contours = []
    masks = []
    # handle `/` missing
    if path[-1] != '/': path += '/'
    # get contour file
    contour_file = get_contour_file(path)
    # get slice orders
    ordered_slices = slice_order(path)
    # get contour dict
    print(contour_file, path, str(index))
    contour_dict = get_contour_dict(contour_file, path, index)
    
    for k,v in ordered_slices:
        # get data from contour dict
        if k in contour_dict:
            images.append(contour_dict[k][0])
            contours.append(contour_dict[k][1])
            masks.append(contour_dict[k][2])
            #masks.append(sni.binary_fill_holes(contour_dict[k][1]).astype(int))
            # get data from dicom.read_file
        else:
            img_arr = dicom.read_file(path + k + '.dcm').pixel_array
            contour_arr = np.zeros_like(img_arr)
            images.append(img_arr)
            contours.append(contour_arr)
            masks.append(contour_arr)
            
    return np.array(images), np.array(contours), np.array(masks)

def paste_from_ct(ct_file, image, mask):
    ct_image = sitk.ReadImage(ct_file)
    ct_arr = sitk.GetArrayFromImage(ct_image)
    image_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)
    mask_idx = np.where(mask_arr == 0)
    image_arr[mask_idx] = ct_arr[mask_idx]
    new_image = sitk.GetImageFromArray(image_arr)
    new_image.CopyInformation(image)
    return new_image

def process_image(I_image, image_case, image_folder):
    intensity_filter = sitk.IntensityWindowingImageFilter()
    intensity_filter.SetOutputMinimum(0.0)
    intensity_filter.SetOutputMaximum(1.0)
    if image_case == '00':
        intensity_filter.SetWindowMaximum(300)
        intensity_filter.SetWindowMinimum(-150)
    else:    
        intensity_filter.SetWindowMaximum(600)
        intensity_filter.SetWindowMinimum(-800)
    I_image = intensity_filter.Execute(I_image)
 
#    I_mask_high = sitk.Cast(sitk.BinaryThreshold(I_image, lowerThreshold=0.0, upperThreshold=1750, insideValue=1, outsideValue=0), sitk.sitkUInt8)   
    I_mask = sitk.Cast(sitk.BinaryThreshold(I_image, lowerThreshold=0.05, upperThreshold=0.95, insideValue=1, outsideValue=0), sitk.sitkUInt8)
#    I_mask = sitk.Cast(sitk.Multiply(I_mask_high, I_mask_low), sitk.sitkUInt8)
    
    sitk.WriteImage(I_image, os.path.join(image_folder, "19"+image_case+"-01__Studies_image_3D.nii.gz"))
    #sitk.WriteImage(I_mask, os.path.join(image_folder, "19"+image_case+"-01__Studies_image_mask.nii.gz"))

    #if image_case != '00':
    #    ct_image_file = os.path.join(image_folder, '1900-01__Studies_image_3D.nii.gz')
    #    I_image = paste_from_ct(ct_image_file, I_image, I_mask)
    #    sitk.WriteImage(I_image, os.path.join(image_folder, "19"+image_case+"-01__Studies_image_pasted.nii.gz"))
    return I_image, I_mask


def get_seg_from_contour(dicom_folder, roi_index, image, name, image_folder, image_case):
    _, contour, mask = get_data(dicom_folder, index=roi_index)
    I_contour = sitk.GetImageFromArray(contour)
    I_mask = sitk.GetImageFromArray(mask)    
    I_contour.CopyInformation(image)
    I_mask.CopyInformation(image)

    #sitk.WriteImage(I_contour, os.path.join(image_folder, "19"+image_case+"-01__Studies_"+name+"_contour.nii.gz"))
    sitk.WriteImage(I_mask, os.path.join(image_folder, "19"+image_case+"-01__Studies_"+name+"_label.nii.gz"))
    return I_contour, I_mask

def old_get_weighted_mask(I_mask_ld, I_mask_hd, image_folder, image_case):
    I_ld_arr = sitk.GetArrayFromImage(I_mask_ld)
    I_hd_arr = sitk.GetArrayFromImage(I_mask_hd)
    
    I_arr =  np.zeros_like(I_ld_arr, dtype=np.float32) 
    ld_idx = np.where(I_ld_arr == 1)
    hd_idx = np.where(I_hd_arr == 1)  
    I_arr[ld_idx] = 1.0
    I_arr[hd_idx] = 1.0

    weighted_mask = sitk.GetImageFromArray(I_arr)
    weighted_mask.CopyInformation(I_mask_ld)

    smooth_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    smooth_filter.SetSigma([5,5,5])
    weighted_mask = smooth_filter.Execute(weighted_mask)

    sitk.WriteImage(weighted_mask, os.path.join(image_folder, "19"+image_case+"-01__Studies_weighted_mask.nii.gz"))

    return



def get_seg_for_patient_case(patient, image_case):
    image_folder = os.path.join(image_root_folder, patient)
    raw_image_folder = os.path.join(image_folder, 'raw_images')
    os.system('mkdir {}'.format(raw_image_folder))
    dicom_folder = os.path.join(dicom_root_folder, patient)
    dicom_case_folder = os.path.join(dicom_folder, '19{}-01__Studies'.format(image_case))
    contour_file = get_contour_file(dicom_case_folder)

    I_image = sitk.ReadImage(os.path.join(image_folder, "19{}-01__Studies.nii".format(image_case)))
    sitk.WriteImage(I_image, os.path.join(raw_image_folder, "19{}-01__Studies_image_3D.nii.gz".format(image_case)))

    if contour_file is not None:
        contour_data = pydicom.read_file(os.path.join(dicom_case_folder, contour_file))
        roi_names = get_roi_names(contour_data)
        print(roi_names)
        labels_csv = pd.read_csv('data.csv', header=None, dtype=str)
        num_of_images = len(labels_csv)
        for idx in range(num_of_images):
            c_patient = labels_csv.iloc[idx, 0]
            c_case = labels_csv.iloc[idx, 1]
            if c_patient == patient:
                if c_case == image_case:
                    roi_smbowel = int(labels_csv.iloc[idx, 2])
                    roi_stomachduo = int(labels_csv.iloc[idx, 3])
                    if image_case == '00':
                       roi_lowptv = int(labels_csv.iloc[idx, 4])
                       roi_highptv = int(labels_csv.iloc[idx, 5])
                    break
        if 'roi_smbowel' not in locals():
            roi_smbowel = int(input("Please selct index for small bowel for image_case {}:".format(image_case)))
        if 'roi_stomachduo' not in locals():
            roi_stomachduo = int(input("Please selct index for StamachDuo for image_case {}:".format(image_case)))


        I_contours = []
        I_masks = []
        I_contour_sb, I_mask_sb = get_seg_from_contour(dicom_case_folder, roi_smbowel, I_image, 'SmBowel', raw_image_folder, image_case)
        I_contour_sd, I_mask_sd = get_seg_from_contour(dicom_case_folder, roi_stomachduo, I_image, 'StomachDuo', raw_image_folder, image_case)
 
        #I_contours.append(sitk.Cast(I_contour_sb, sitk.sitkLabelUInt8))
        #I_contours.append(sitk.Cast(I_contour_sd, sitk.sitkLabelUInt8))
        #I_masks.append(sitk.Cast(I_mask_sb, sitk.sitkLabelUInt8))
        #I_masks.append(sitk.Cast(I_mask_sd, sitk.sitkLabelUInt8))

        #merge_filter = sitk.MergeLabelMapFilter()
        #merge_filter.SetMethod(Method=0)

        #I_mask_all = sitk.Cast(merge_filter.Execute(I_masks), sitk.sitkUInt8)
        #sitk.WriteImage(I_mask_all, os.path.join(raw_image_folder, "19"+image_case+"-01__Studies_label.nii.gz"))

        #I_contours_dose = [] 
        if image_case == '00':
            if 'roi_lowptv' not in locals():
               roi_lowptv = int(input("Please selct index for Low PTV for image_case {}:".format(image_case)))
            if 'roi_highptv' not in locals():
               roi_highptv = int(input("Please selct index for High PTV for image_case {}:".format(image_case)))
            I_contour_ld, I_mask_ld = get_seg_from_contour(dicom_case_folder, roi_lowptv, I_image, 'LowPTV', raw_image_folder, image_case)
            I_contour_hd, I_mask_hd = get_seg_from_contour(dicom_case_folder, roi_highptv, I_image, 'HighPTV', raw_image_folder, image_case)

            #I_contours.append(sitk.Cast(I_contour_ld, sitk.sitkLabelUInt8))
            #I_contours.append(sitk.Cast(I_contour_hd, sitk.sitkLabelUInt8))
            
            #I_contours_dose.append(sitk.Cast(I_contour_ld, sitk.sitkLabelUInt8))
            #I_contours_dose.append(sitk.Cast(I_contour_hd, sitk.sitkLabelUInt8))

            #I_contour_dose = sitk.Cast(merge_filter.Execute(I_contours_dose), sitk.sitkUInt8)
            #sitk.WriteImage(I_contour_dose, os.path.join(image_folder, "19"+image_case+"-01__Studies_Dose_contour.nii.gz"))
            # I_masks.append(sitk.Cast(I_mask_ld, sitk.sitkLabelUInt8))
            # I_masks.append(sitk.Cast(I_mask_hd, sitk.sitkLabelUInt8))           

            #get_weighted_mask(I_mask_ld, I_mask_hd, image_folder, image_case)
 
        #else:
            #I_contour_dose = sitk.ReadImage(os.path.join(image_folder, "1900-01__Studies_Dose_contour.nii.gz"))
            #I_contour_dose.CopyInformation(I_image)
            #I_contours.append(sitk.Cast(I_contour_dose, sitk.sitkLabelUInt8))
       
        #I_contour_all = sitk.Cast(merge_filter.Execute(I_contours), sitk.sitkUInt8)
 
        #sitk.WriteImage(I_contour_all, os.path.join(image_folder, "19"+image_case+"-01__Studies_contour.nii.gz"))
      
    #I_image, I_mask = process_image(I_image, image_case, image_folder)

    return



if __name__ == '__main__':
    patient = sys.argv[1]
    case = sys.argv[2]
    get_seg_for_patient_case(patient, case)

