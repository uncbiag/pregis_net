import SimpleITK as sitk
import os
import glob
import numpy as np


def bbox2(img):
    rows = np.any(img, axis=(1,2))
    cols = np.any(img, axis=(0,2))
    deps = np.any(img, axis=(0,1))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    dmin, dmax = np.where(deps)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, dmin, dmax


def check_folder(image_folder):
    patients = glob.glob(os.path.join(image_folder, '1822*'))
    for patient_folder in patients:
        cropped = glob.glob(os.path.join(patient_folder, "*cropped_images"))
        for image_folder in cropped:
            image_file = os.path.join(image_folder, '1900-01__Studies_weighted_mask_before_smooth.nii.gz')

            # image_case = image_folder.split("-01_cropped_images")[0][-2:]  # last two digit before -01_cropped_images
            # image_file = os.path.join(image_folder, '19{}-01__Studies_image_normalized.nii.gz'.format(image_case))

            image = sitk.ReadImage(image_file)
            print(image.GetSize())
            img = sitk.GetArrayFromImage(image)
            #rmin, rmax, cmin, cmax, dmin, dmax = bbox2(img)
            #print(rmin, cmin, dmin, rmax-rmin, cmax-cmin, dmax-dmin)
            shape_filter = sitk.LabelShapeStatisticsImageFilter()
            shape_filter.Execute(sitk.Cast(image, sitk.sitkUInt8))
            box = np.array(shape_filter.GetBoundingBox(1), ndmin=2)
            print(box[0])


if __name__ == '__main__':
    folder = "/playpen1/xhs400/Research/data/r21/data/ct-cbct/images"
    with_contours = os.path.join(folder, "with_cbct_contours")
    without_contours = os.path.join(folder, "without_cbct_contours")
    check_folder(with_contours)
    check_folder(without_contours)
