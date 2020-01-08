import sys
import os
import os.path
import subprocess
import SimpleITK as sitk
sys.path.insert(1, 'all_steps')


from global_variables import *




if __name__ == "__main__":
    patient = sys.argv[1]
    process_dicom_image(patient)
