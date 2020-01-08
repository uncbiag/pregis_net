import SimpleITK as sitk
import glob
import os
import sys

from global_variables import *

def resample(patient, image_case):
    patient_folder = os.path.join(image_root_folder, patient)
    raw_folder = os.path.join(patient_folder, 'raw_images')
    resampled_folder = os.path.join(patient_folder, 'resampled_images')
    os.system('mkdir {}'.format(resampled_folder))

    image_file = os.path.join(raw_folder, '19{}-01__Studies_image_3D.nii.gz'.format(image_case))
    main_image = sitk.ReadImage(image_file)
    origin = main_image.GetOrigin()
    spacing = main_image.GetSpacing()
    sz = main_image.GetSize()
    physical_size = [int(a*b) for a,b in zip(spacing, sz)]

    print(origin)
    print(spacing)
    print(sz)
    print(physical_size)

    all_images = glob.glob(os.path.join(raw_folder, '19{}-01__Studies*'.format(image_case)))

    resampleFilter = sitk.ResampleImageFilter()
    #resampleFilter.SetInterpolator(sitk.sitkLinear)
    resampleFilter.SetOutputOrigin(origin)
    resampleFilter.SetSize(tuple(physical_size))

    for image_file in all_images:
        print("Resampling {}".format(image_file))
        image = sitk.ReadImage(image_file)
        minmaxFilter = sitk.MinimumMaximumImageFilter()
        minmaxFilter.Execute(image)
        resampleFilter.SetDefaultPixelValue(minmaxFilter.GetMinimum())

        image_name = os.path.basename(image_file)
        resampled_image_file = os.path.join(resampled_folder, image_name)
        if "label" in image_file:
            resampleFilter.SetInterpolator(sitk.sitkNearestNeighbor)
            resampled_image = resampleFilter.Execute(image)
        else:
            resampleFilter.SetInterpolator(sitk.sitkLinear)
            resampled_image = resampleFilter.Execute(image)
        resampled_image.SetOrigin([0,0,0])
        sitk.WriteImage(resampled_image, resampled_image_file)
    

if __name__ == "__main__":
    patient = sys.argv[1]
    image_case = sys.argv[2]

    resample(patient, image_case)



