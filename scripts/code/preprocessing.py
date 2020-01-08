import sys
import os
import subprocess
import SimpleITK as sitk
import glob
import pandas as pd
import numpy as np
import cv2
import argparse
from scipy import ndimage
import datetime


class ProcessR21(object):

    def __init__(self, with_cbct_contours):
        self.data_folder = "/playpen1/xhs400/Research/data/r21/data/ct-cbct"
        self.with_cbct_contours = with_cbct_contours
        if self.with_cbct_contours:
            self.dicom_root_folder = os.path.join(self.data_folder, 'dicoms/with_cbct_contours')
            self.image_root_folder = os.path.join(self.data_folder, 'images/with_cbct_contours')
            self.zip_root_folder = os.path.join(self.data_folder, 'zips/with_cbct_contours')
        else:
            self.dicom_root_folder = os.path.join(self.data_folder, 'dicoms/without_cbct_contours')
            self.zip_root_folder = os.path.join(self.data_folder, 'zips/without_cbct_contours')
            self.image_root_folder = os.path.join(self.data_folder, 'images/without_cbct_contours')
        return

    def process(self, patient):
        # self.process_dicom_image(patient)
        image_cases = sorted(self.get_image_cases(patient))
        print(image_cases)
        if len(image_cases) == 0:
            print("No image case found! Exit")
            return
        for image_case in image_cases:
            #break
            self.get_seg_for_patient_case(patient, image_case)
            self.resample(patient, image_case)
            self.normalize_image(patient, image_case)
            if image_case == '00':
                self.dilate_lowPTV(patient)
            #if self.with_cbct_contours or image_case == '00':
            #    self.process_labels_to_contour(patient, image_case)

        self.crop_image_roi(patient, image_cases)
        for image_case in image_cases:
            if self.with_cbct_contours or image_case == '00':
                self.process_labels_to_contour(patient, image_case)
        #self.get_roi_slices(patient, image_cases)
        #self.get_roi_images(patient, image_cases)

        #self.reg_model = "lddmm"
        #self.mermaid_registration(patient, image_cases)
        return

    def crop_image_roi(self, patient, image_cases):
        patient_folder = os.path.join(self.image_root_folder, patient)
        roi_file = os.path.join(patient_folder, 'Resampled', 'planCT_OG', 'ROI_40_label.nii.gz')
        roi_itk = sitk.ReadImage(roi_file)
        label_filter = sitk.LabelShapeStatisticsImageFilter()
        label_filter.Execute(roi_itk)
        bbox = label_filter.GetBoundingBox(1)
        print(bbox)
        for image_case in image_cases:
            if image_case == '00':
                resampled_folder = os.path.join(patient_folder, 'Resampled', 'planCT_OG')
                cropped_folder = os.path.join(patient_folder, 'Cropped', 'planCT_OG')
            else:
                resampled_folder = os.path.join(patient_folder, 'Resampled', 'CBCT{}_OG'.format(image_case))
                cropped_folder = os.path.join(patient_folder, 'Cropped', 'CBCT{}_OG'.format(image_case))
            os.system('mkdir -p ' + cropped_folder)
            all_images = glob.glob(os.path.join(resampled_folder, '*.nii.gz'))
            for image_file in all_images:
                print("Cropping {}".format(image_file))
                image_itk = sitk.ReadImage(image_file)
                image_name = os.path.basename(image_file)
                cropped_image = image_itk[bbox[0]:bbox[0]+bbox[3], bbox[1]:bbox[1]+bbox[4], bbox[2]:bbox[2]+bbox[5]]
                cropped_file = os.path.join(cropped_folder, image_name)
                sitk.WriteImage(cropped_image, cropped_file)
        return

    def mermaid_registration(self, patient, image_cases):
        for image_case in image_cases:
            if image_case == '00':
                continue
            reg_folder = os.path.join(self.image_root_folder, patient, '00_'+image_case+'_reg_folder')
            now = datetime.datetime.now()
            my_time = "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
            result_folder = os.path.join(reg_folder, 'mermaid_' + my_time)
            os.system('mkdir ' + result_folder)

            moving_image_file = os.path.join(reg_folder, '1900-01__Normalized.nii.gz')
            target_image_file = os.path.join(reg_folder, '19{}-01__Normalized.nii.gz'.format(image_case))
            result_image_file = os.path.join(result_folder, '00_{}__warped_image.nii.gz'.format(image_case))
            forward_disp_file = os.path.join(result_folder, '00_{}__forward_disp.nii.gz'.format(image_case))
            inverse_disp_file = os.path.join(result_folder, '00_{}__inverse_disp.nii.gz'.format(image_case))
            momentum_file = os.path.join(result_folder, '00_{}__momentum.npy'.format(image_case))
            moving_sb_label_file = os.path.join(reg_folder, '1900-01__SmBowel.nii.gz')
            moving_sd_label_file = os.path.join(reg_folder, '1900-01__StomachDuo.nii.gz')
            warped_sb_label_file = os.path.join(result_folder, '00_{}__warped_SmBowel.nii.gz'.format(image_case))
            warped_sd_label_file = os.path.join(result_folder, '00_{}__warped_StomachDuo.nii.gz'.format(image_case))
            output_file = os.path.join(result_folder, '00_{}__output.txt'.format(image_case))

            if self.reg_model == 'lddmm':
                json_file = 'mermaid_config_lddmm.json'
            elif self.reg_model == 'svf':
                json_file = 'mermaid_config_svf.json'
            else:
                raise ValueError("registration model not supported")

            cmd = "python -W ignore mermaid_reg.py --moving {} --target {} --warped {} --disp {} --inv {} --momentum {} --json {} --labels {} {} --warped_labels {} {} >> {}".format(moving_image_file, target_image_file, result_image_file, forward_disp_file, inverse_disp_file, momentum_file, json_file, moving_sb_label_file, moving_sd_label_file, warped_sb_label_file, warped_sd_label_file, output_file)
            process = subprocess.Popen(cmd, shell=True)
            process.wait()

           
    def get_roi_slices(self, patient, image_cases):
        roi_file = os.path.join(self.image_root_folder, patient, '1900-01__Resampled', 'ROI_40.nii.gz')
        roi = sitk.ReadImage(roi_file)
        roi_arr = sitk.GetArrayFromImage(roi)
        slice_folder = os.path.join(self.image_root_folder, patient, '2D_slice')
        os.system('mkdir ' + slice_folder)
        image_arrs = {}
        mask_arrs = {}
        sb_label_arrs = {}
        sd_label_arrs = {}
        for image_case in image_cases:
            image_file = os.path.join(self.image_root_folder, patient, '19' + image_case + '-01__Resampled',
                                      '19' + image_case + '-01__Normalized.nii.gz')
            image = sitk.ReadImage(image_file)
            image_arr = sitk.GetArrayFromImage(image)
            image_arrs[image_case] = image_arr

            if self.with_cbct_contours:
                sb_label_file = os.path.join(self.image_root_folder, patient, '19' + image_case + '-01__Resampled', 'SmBowel.nii.gz')
                sd_label_file = os.path.join(self.image_root_folder, patient, '19' + image_case + '-01__Resampled', 'StomachDuo.nii.gz')
                sb_label = sitk.ReadImage(sb_label_file)
                sd_label = sitk.ReadImage(sd_label_file)
                sb_label_arr = sitk.GetArrayFromImage(sb_label)
                sd_label_arr = sitk.GetArrayFromImage(sd_label)
                sb_label_arrs[image_case] = sb_label_arr
                sd_label_arrs[image_case] = sd_label_arr
              
            if image_case != '00':
                mask_file = os.path.join(self.image_root_folder, patient, '19' + image_case + '-01__Resampled',
                                         '19' + image_case + '-01__Mask.nii.gz')
                mask = sitk.ReadImage(mask_file)
                mask_arr = sitk.GetArrayFromImage(mask)
                mask_arrs[image_case] = mask_arr

        slices_dict = self.get_cropped_slice(image_arrs, roi_arr)
        for image_case, slices in slices_dict.items():
            print("{} has {} slices".format(image_case, len(slices)))
            for slice_index in range(len(slices)):
                slice_to_save = os.path.join(slice_folder, '19{}-01_slice_{}.nii.gz'.format(image_case, slice_index))
                sitk.WriteImage(sitk.GetImageFromArray(slices[slice_index]), slice_to_save)
        mask_dict = self.get_cropped_slice(mask_arrs, roi_arr)
        for image_case, masks in mask_dict.items():
            print("{} has {} slices".format(image_case, len(masks)))
            for mask_index in range(len(masks)):
                mask_to_save = os.path.join(slice_folder, '19{}-01_mask_{}.nii.gz'.format(image_case, mask_index))
                sitk.WriteImage(sitk.GetImageFromArray(masks[mask_index]), mask_to_save)
        if self.with_cbct_contours:
            sb_dict = self.get_cropped_slice(sb_label_arrs, roi_arr)
            sd_dict = self.get_cropped_slice(sd_label_arrs, roi_arr)
            for image_case, sb_labels in sb_dict.items():
                for sb_index in range(len(sb_labels)):
                    sb_to_save = os.path.join(slice_folder, '19{}-01_SmBowel_{}.nii.gz'.format(image_case, sb_index))
                    sitk.WriteImage(sitk.GetImageFromArray(sb_labels[sb_index]), sb_to_save)
            for image_case, sd_labels in sd_dict.items():
                for sd_index in range(len(sd_labels)):
                    sd_to_save = os.path.join(slice_folder, '19{}-01_StomachDuo_{}.nii.gz'.format(image_case, sb_index))
                    sitk.WriteImage(sitk.GetImageFromArray(sd_labels[sd_index]), sd_to_save)

        return

    def get_roi_images(self, patient, image_cases):
        patient_folder = os.path.join(self.image_root_folder, patient)
        ct_folder = os.path.join(patient_folder, '1900-01__Resampled')
        roi_file = os.path.join(ct_folder, 'ROI_40.nii.gz')
        roi = sitk.ReadImage(roi_file)
        roi_arr = sitk.GetArrayFromImage(roi)
        c_of_m_3D = ndimage.measurements.center_of_mass(roi_arr)
        center_of_mass = [c_of_m_3D[1], c_of_m_3D[2]]
        sz_z, sz_x, sz_y = roi_arr.shape
        top_z = None
        bottom_z = None
        for z in range(sz_z):
            roi_slice = roi_arr[z, :, :]
            if np.count_nonzero(roi_slice) == 0:
                if top_z is not None and bottom_z is None:
                    bottom_z = z
            if np.count_nonzero(roi_slice) > 0:
                if top_z is None:
                    top_z = z
            if top_z is not None and bottom_z is not None:
                break
        print(top_z, bottom_z)
        
        for image_case in image_cases:
            if image_case == '00':
                continue
            case_folder = os.path.join(patient_folder, '19'+image_case+'-01__Resampled')
            cbct_mask = os.path.join(case_folder, '19'+image_case+'-01__Mask.nii.gz')
            cbct_mask_itk = sitk.ReadImage(cbct_mask)
            cbct_mask_arr = sitk.GetArrayFromImage(cbct_mask_itk)
            maskout_indices = (cbct_mask_arr == 0)
            registration_folder = os.path.join(patient_folder, '00_' + image_case + '_reg_folder')
            os.system('mkdir ' + registration_folder)
            ct_images = glob.glob(os.path.join(ct_folder, '*.nii.gz'))
            for ct_image in ct_images:
                name = os.path.basename(ct_image)
                if "Studies" in name:
                    continue
                ct_image_itk = sitk.ReadImage(ct_image)
                ct_image_arr = sitk.GetArrayFromImage(ct_image_itk)
                if "Normalized" in name:
                    ct_image_arr[maskout_indices] = -1
                else:
                    ct_image_arr[maskout_indices] = 0
                expanded_image_arr = np.pad(ct_image_arr, ((0,0), (50,50), (50, 50)), mode='edge')
                cropped_arr = expanded_image_arr[top_z:bottom_z,
                        int(center_of_mass[0]) - 96 + 50:int(center_of_mass[0]) + 96 + 50,
                        int(center_of_mass[1]) - 96 + 50:int(center_of_mass[1]) + 96 + 50
                        ]
                cropped_itk = sitk.GetImageFromArray(cropped_arr)
                cropped_itk.SetSpacing(ct_image_itk.GetSpacing())
                if "1900-01" in name:
                    result_image = os.path.join(registration_folder, name)
                else:
                    result_image = os.path.join(registration_folder, "1900-01__" + name)
                sitk.WriteImage(cropped_itk, result_image)

            cb_images = glob.glob(os.path.join(case_folder, '*.nii.gz'))
            for cb_image in cb_images:
                name = os.path.basename(cb_image)
                if "Studies" in name:
                    continue
                cb_image_itk = sitk.ReadImage(cb_image)
                cb_image_arr = sitk.GetArrayFromImage(cb_image_itk)
                expanded_image_arr = np.pad(cb_image_arr, ((0,0), (50,50), (50,50)), mode='edge')
                cropped_arr = expanded_image_arr[top_z:bottom_z,
                        int(center_of_mass[0]) - 96 + 50:int(center_of_mass[0]) + 96 + 50,
                        int(center_of_mass[1]) - 96 + 50:int(center_of_mass[1]) + 96 + 50
                        ]
                cropped_itk = sitk.GetImageFromArray(cropped_arr)
                cropped_itk.SetSpacing(cb_image_itk.GetSpacing())
                if "19"+image_case+"-01" in name:
                    result_image = os.path.join(registration_folder, name)
                else:
                    result_image = os.path.join(registration_folder, "19"+image_case+"-01__"+name)
                sitk.WriteImage(cropped_itk, result_image)
 

    def get_cropped_slice(self, image_arrs, roi_arr):
        c_of_m_3D = ndimage.measurements.center_of_mass(roi_arr)
        print(c_of_m_3D)
        center_of_mass = [c_of_m_3D[1], c_of_m_3D[2]]
        sz_z, sz_x, sz_y = roi_arr.shape
        slices_dict = {}
        for key in image_arrs:
            slices_dict[key] = []
        for z in range(sz_z):
            roi_slice = roi_arr[z, :, :]
            if np.count_nonzero(roi_slice) == 0:
                continue
            for key, image_arr in image_arrs.items():
                image_slice = np.pad(image_arr[z, :, :], 50, mode='edge')
                cropped_slice = image_slice[
                                int(center_of_mass[0]) - 96 + 50:int(center_of_mass[0]) + 96 + 50,
                                int(center_of_mass[1]) - 96 + 50:int(center_of_mass[1]) + 96 + 50
                                ]
                slices_dict[key].append(cropped_slice)
        return slices_dict


    def process_labels_to_contour(self, patient, image_case):
        if image_case == '00':
            cropped_folder = os.path.join(self.image_root_folder, patient, "Cropped", "planCT_OG")
        else:
            cropped_folder = os.path.join(self.image_root_folder, patient, "Cropped", "CBCT{}_OG".format(image_case))
        I_contours = []
        I_contours.append(
            sitk.Cast(self.get_contour_from_label(patient, image_case, 'StomachDuo'), sitk.sitkLabelUInt8))
        I_contours.append(sitk.Cast(self.get_contour_from_label(patient, image_case, 'SmBowel'), sitk.sitkLabelUInt8))
        if image_case == '00':
            self.get_contour_from_label(patient, image_case, 'HighPTV')
            self.get_contour_from_label(patient, image_case, 'LowPTV')
            self.get_contour_from_label(patient, image_case, 'ROI_10')

        merge_filter = sitk.MergeLabelMapFilter()
        merge_filter.SetMethod(Method=0)

        I_contour_all = sitk.Cast(merge_filter.Execute(I_contours), sitk.sitkUInt8)
        sitk.WriteImage(I_contour_all, os.path.join(cropped_folder, 'all_contours.nii.gz'.format(image_case)))
        I_labels = []
        I_labels.append(sitk.Cast(sitk.ReadImage(os.path.join(cropped_folder, 'SmBowel_label.nii.gz')), sitk.sitkLabelUInt8))
        I_labels.append(sitk.Cast(sitk.ReadImage(os.path.join(cropped_folder, 'StomachDuo_label.nii.gz')), sitk.sitkLabelUInt8))
        I_label_all = sitk.Cast(merge_filter.Execute(I_labels), sitk.sitkUInt8)
        sitk.WriteImage(I_label_all, os.path.join(cropped_folder, 'all_labels.nii.gz'))
        return

    def get_contour_from_label(self, patient, image_case, label_name):
        if image_case == '00':
            cropped_folder = os.path.join(self.image_root_folder, patient, 'Cropped', 'planCT_OG')
        else:
            cropped_folder = os.path.join(self.image_root_folder, patient, 'Cropped', 'CBCT{}_OG'.format(image_case))
        label_file = os.path.join(cropped_folder, '{}_label.nii.gz'.format(label_name))
        I_label = sitk.Cast(sitk.ReadImage(label_file), sitk.sitkUInt8)
        I_label_arr = sitk.GetArrayFromImage(I_label)
        z, x, y = I_label_arr.shape
        I_contour_arr = np.zeros_like(I_label_arr)

        for i in range(z):
            label_slice = I_label_arr[i, :, :]
            contour_slice = np.zeros_like(label_slice, np.uint8)
            if np.sum(label_slice) > 0:
                _, contours, _ = cv2.findContours(label_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.polylines(contour_slice, contours, isClosed=True, color=1)
                I_contour_arr[i] = contour_slice

        I_contour = sitk.GetImageFromArray(I_contour_arr)
        I_contour.CopyInformation(I_label)
        contour_file = os.path.join(cropped_folder,
                                    '{}_contour.nii.gz'.format(label_name))
        print("Writing {}".format(contour_file))
        sitk.WriteImage(I_contour, contour_file)
        return I_contour

    def dilate_lowPTV(self, patient):
        patient_folder = os.path.join(self.image_root_folder, patient)
        resampled_folder = os.path.join(patient_folder, 'Resampled', 'planCT_OG')
        print("Dilate Low PTV Region to ROI")
        # dilate lowPTV 10mm
        lowPTV_file = os.path.join(resampled_folder, 'LowPTV_label.nii.gz')
        lowPTV = sitk.ReadImage(lowPTV_file)
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius([10, 10, 10])
        roi = dilate_filter.Execute(lowPTV)
        roi_file = os.path.join(resampled_folder, 'ROI_10_label.nii.gz')
        sitk.WriteImage(roi, roi_file)
        dilate_filter.SetKernelRadius([40, 40, 40])
        roi = dilate_filter.Execute(lowPTV)
        roi_file = os.path.join(resampled_folder, 'ROI_40_label.nii.gz')
        sitk.WriteImage(roi, roi_file)
        os.system('rm ' + os.path.join(resampled_folder, 'ROI.nii.gz'))
        os.system('rm ' + os.path.join(resampled_folder, 'ROI_contour.nii.gz'))
        return

    def normalize_image(self, patient, image_case):
        print("Normalizing Image {} {}".format(patient, image_case))
        patient_folder = os.path.join(self.image_root_folder, patient)
        if image_case == '00':
            resample_folder = os.path.join(patient_folder, "Resampled", "planCT_OG")
        else:
            resample_folder = os.path.join(patient_folder, "Resampled", "CBCT{}_OG".format(image_case))
        image_file = os.path.join(resample_folder, 'image.nii.gz')
        I_image = sitk.Cast(sitk.ReadImage(image_file), sitk.sitkFloat32)

        intensity_filter = sitk.IntensityWindowingImageFilter()
        intensity_filter.SetOutputMinimum(-1.0)
        intensity_filter.SetOutputMaximum(1.0)

        data_csv = pd.read_csv('data.csv', header=None, dtype=str)
        num_of_images = len(data_csv)
        for idx in range(num_of_images):
            c_patient = data_csv.iloc[idx, 0]
            c_case = data_csv.iloc[idx, 1]
            if c_patient == patient and c_case == image_case:
                windowMinimum = data_csv.iloc[idx, 6]
                windowMaximum = data_csv.iloc[idx, 7]
                break

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
        I_mask = sitk.Cast(
            sitk.BinaryThreshold(I_image, lowerThreshold=-0.99, upperThreshold=1, insideValue=1, outsideValue=0),
            sitk.sitkUInt8
        )

        fillhole_filter = sitk.BinaryFillholeImageFilter()
        I_mask = fillhole_filter.Execute(I_mask)

        image_norm_file = os.path.join(resample_folder, 'image_normalized.nii.gz')
        sitk.WriteImage(I_image, image_norm_file)
        mask_file = os.path.join(resample_folder, 'image_mask.nii.gz')
        sitk.WriteImage(I_mask, mask_file)

    def resample(self, patient, image_case):
        patient_folder = os.path.join(self.image_root_folder, patient)
        if image_case == '00':
            image_case_folder = os.path.join(patient_folder, 'Origin', 'planCT_OG')
            resampled_folder = os.path.join(patient_folder, 'Resampled', 'planCT_OG')
        else:
            image_case_folder = os.path.join(patient_folder, 'Origin', 'CBCT{}_OG'.format(image_case))
            resampled_folder = os.path.join(patient_folder, 'Resampled', 'CBCT{}_OG'.format(image_case))
        os.system('mkdir -p {}'.format(resampled_folder))

        image_file = os.path.join(image_case_folder, 'image.nii.gz')
        main_image = sitk.ReadImage(image_file)
        origin = main_image.GetOrigin()
        spacing = main_image.GetSpacing()
        sz = main_image.GetSize()
        physical_size = [round(a * b) for a, b in zip(spacing, sz)]

        all_images = glob.glob(os.path.join(image_case_folder, '*.nii.gz'))

        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetOutputOrigin(origin)
        new_spacing = [1, 1, 1]
        new_size = physical_size
        resampleFilter.SetSize(tuple(new_size))
        resampleFilter.SetOutputSpacing(tuple(new_spacing))

        for image_file in all_images:
            print("Resampling {}".format(image_file))
            image = sitk.ReadImage(image_file)
            minmaxFilter = sitk.MinimumMaximumImageFilter()
            minmaxFilter.Execute(image)
            resampleFilter.SetDefaultPixelValue(minmaxFilter.GetMinimum())

            image_name = os.path.basename(image_file)
            resampled_image_file = os.path.join(resampled_folder, image_name)
            if "image" in image_file and not "label" in image_file:
                print(image_file, "Linear")
                resampleFilter.SetInterpolator(sitk.sitkLinear)
            else:
                print(image_file, "NearestNeighbor")
                resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
            resampled_image = resampleFilter.Execute(image)
            resampled_image.SetOrigin([0, 0, 0])
            sitk.WriteImage(resampled_image, resampled_image_file)

    def get_seg_for_patient_case(self, patient, image_case):
        if image_case == '00':
            image_case_folder = os.path.join(self.image_root_folder, patient, 'Origin', 'planCT_OG')
        else:
            image_case_folder = os.path.join(self.image_root_folder, patient, 'Origin', 'CBCT{}_OG'.format(image_case)) 
        os.system('mkdir -p ' + image_case_folder)
        dicom_case_folder = os.path.join(self.dicom_root_folder, patient, '19' + image_case + '-01__Studies')
        cmd = "plastimatch convert " \
              "--input {} " \
              "--output-img {} " \
              "--output-prefix {} " \
              "--prefix-format nii.gz".format(
            dicom_case_folder,
            os.path.join(image_case_folder, 'image.nii.gz'),
            image_case_folder,
            image_case_folder
        )
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

        if not image_case == '00' and self.with_cbct_contours is not True:
            return
        # rename label files, adjust intensity level for ct/cbct image
        labels_csv = pd.read_csv('data.csv', header=None, dtype=str)
        num_of_images = len(labels_csv)
        for idx in range(num_of_images):
            c_patient = labels_csv.iloc[idx, 0]
            c_case = labels_csv.iloc[idx, 1]
            if c_patient == patient and c_case == image_case:
                roi_smbowel = labels_csv.iloc[idx, 2]
                roi_stomachduo = labels_csv.iloc[idx, 3]
                os.system('mv ' + os.path.join(image_case_folder, roi_smbowel + '.nii.gz')
                          + ' ' + os.path.join(image_case_folder, 'SmBowel_label.nii.gz'))
                print('mv ' + os.path.join(image_case_folder, roi_smbowel + '.nii.gz')
                      + ' ' + os.path.join(image_case_folder, 'SmBowel_label.nii.gz'))
                os.system('mv ' + os.path.join(image_case_folder, roi_stomachduo + '.nii.gz')
                          + ' ' + os.path.join(image_case_folder, 'StomachDuo_label.nii.gz'))
                print('mv ' + os.path.join(image_case_folder, roi_stomachduo + '.nii.gz')
                      + ' ' + os.path.join(image_case_folder, 'StomachDuo_label.nii.gz'))
                if image_case == '00':
                    roi_lowptv = labels_csv.iloc[idx, 4]
                    roi_highptv = labels_csv.iloc[idx, 5]
                    os.system('mv ' + os.path.join(image_case_folder, roi_lowptv + '.nii.gz')
                              + ' ' + os.path.join(image_case_folder, 'LowPTV_label.nii.gz'))
                    print('mv ' + os.path.join(image_case_folder, roi_lowptv + '.nii.gz')
                          + ' ' + os.path.join(image_case_folder, 'LowPTV_label.nii.gz'))
                    os.system('mv ' + os.path.join(image_case_folder, roi_highptv + '.nii.gz')
                              + ' ' + os.path.join(image_case_folder, 'HighPTV_label.nii.gz'))
                    print('mv ' + os.path.join(image_case_folder, roi_highptv + '.nii.gz')
                          + ' ' + os.path.join(image_case_folder, 'HighPTV_label.nii.gz'))
                break
        return

    def get_image_cases(self, patient):
        image_folder = os.path.join(self.dicom_root_folder, patient)
        images = glob.glob(os.path.join(image_folder, '*-01__Studies'))
        cases = []
        for image in images:
            patient_case = os.path.basename(image)[2:4]
            cases.append(patient_case)
        return cases

    def load_dicoms(self, patient, image_case):
        dicom_folder = os.path.join(self.dicom_root_folder, patient, '19' + image_case + '-01__Studies')
        cmd = ""
        for folder in os.listdir(dicom_folder):
            image_dicom_folder = os.path.join(dicom_folder, folder)
            if os.path.isfile(image_dicom_folder):
                continue
            cmd += "\n" + "mv" + ' "' + image_dicom_folder + '"/*.dcm "' + os.path.join(image_dicom_folder, '../') + '"'
            cmd += "\n" + "rm" + ' ' + "-r" + ' "' + image_dicom_folder + '"'
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

    def process_dicom_image(self, patient):
        zip_file = os.path.join(self.zip_root_folder, patient + '.zip')
        if not os.path.exists(zip_file):
            print("zip file not exist")
            return

        dicom_folder = os.path.join(self.dicom_root_folder, patient)
        image_folder = os.path.join(self.image_root_folder, patient)
        os.system('mkdir -p ' + image_folder)
        os.system('mkdir -p ' + dicom_folder)
        print("Unzipping files.")
        os.system('unzip -qq ' + zip_file + ' ' + '-d' + ' ' + dicom_folder)
        os.system('rm ' + dicom_folder + '/*.txt')
        os.system('rm ' + dicom_folder + '/*.pdf')
        for folder in os.listdir(dicom_folder):
            image_case = folder.split('-')[0][2:]
            self.load_dicoms(patient, image_case)
        return


# def process(patient):
#     process_dicom_image(patient)
#     # print(image_cases)
#     # if len(image_cases) == 0:
#     #     return
#     # for image_case in image_ca
#     #     get_seg_for_patient_case(patient, image_case)
#     #     resample(patient, image_case)
#     #     if image_case == '00':
#     #         dilate_lowPTV(patient)
#     #     normalize_image(patient, image_case)
#     #     if with_cbct_contours or image_case == '00':
#     #         process_labels_to_contour(patient, image_case)
#     #
#     # get_weighted_mask(patient)
#     #
#     # for image_case in image_cases:
#     #     crop_image(patient, image_case)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocessing")
    parser.add_argument('--patients', nargs='+', dest='patients', help='list of patients', type=int)
    # parser.add_argument('--with_contours', dest='with_cbct_contours', action='store_true')
    parser.add_argument('--without_contours', dest='with_cbct_contours', action='store_false')
    parser.set_defaults(with_cbct_contours=True)
    args = parser.parse_args()
    #args = parser.parse_args(["--without_contours"])
    #args = parser.parse_args(['--patients', '6', '9'])
    print(args)
    patients = args.patients
    with_cbct_contours = args.with_cbct_contours
    if with_cbct_contours:
        print("With CBCT Contours")
    else:
        print("Without CBCT Contours")
    preprocesser = ProcessR21(with_cbct_contours)
    if patients is None:
        print("No patient given")
        for i in range(0, 100):
            print("Processing patient 18227{:02d}".format(i))
            preprocesser.process("18227{:02d}".format(i))
    else:
        for patient in patients:
            print("Processing patient 18227{:02d}".format(patient))
            preprocesser.process("18227{:02d}".format(patient))
