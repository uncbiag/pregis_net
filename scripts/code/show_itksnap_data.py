import os
import sys
import subprocess


patient = sys.argv[1]
case = sys.argv[2]


cmd = "itksnap"
root_folder = '/playpen/xhs400/Research/PycharmProjects/r21/data/ct-cbct/images'
patient_folder = os.path.join(root_folder, patient)
case_folder = os.path.join(patient_folder, '19{}-01_processed'.format(case))

ct_image = os.path.join(case_folder, '1900-01__Studies_image_3D.nii.gz')

cb_image = os.path.join(case_folder, '19{}-01__Studies_image_3D.nii.gz'.format(case))
cb_contour = os.path.join(case_folder, '19{}-01__Studies_contour.nii.gz'.format(case))

result_folder = os.path.join(case_folder, 'mermaid_20190823113458/map_0.5')
warped_ct_image = os.path.join(result_folder, '00-{}-warped-image.nii.gz'.format(case))
warped_ct_label = os.path.join(result_folder, '00-{}-warped-two-label.nii.gz'.format(case))
warped_map = os.path.join(result_folder, '00-{}-map.nii.gz'.format(case))
cmd += " -g {} -s {} -o {} {} {} {}".format(ct_image, cb_contour, cb_image, warped_ct_image, warped_ct_label, warped_map)


print(cmd)
process= subprocess.Popen(cmd, shell=True)
process.wait()
