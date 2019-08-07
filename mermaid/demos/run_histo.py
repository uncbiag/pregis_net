import os
import sys
import subprocess

#folder = sys.argv[1]

root_folder = '/playpen/xhs400/Research/data/data_for_pregis_net'
atlas_folder = os.path.join(root_folder, 'atlas_folder')
brats_folder = os.path.join(root_folder, 'brats_affined')
#oasis_folder = os.path.join(root_folder, 'oasis_affined', 'oasis_aff_normed')
cmd = "python normalize_image_intensities.py "

brain_folder = os.path.join(brats_folder, 't1')
output_folder = os.path.join(brats_folder, 'brats_aff_normed')


cmd += ' --desired_output_directory {} '.format(output_folder)
#cmd += ' --dataset_directory_to_compute_cdf {} --suffix_to_compute_cdf {} '.format(oasis_folder, 'nii.gz')
cmd += ' --directory_to_normalize {} --suffix {} '.format(brain_folder, 'nii.gz')
cdf_file = os.path.join(root_folder, 'average_cdf.pt')
#cmd += ' --save_average_cdf_to_file {}'.format(cdf_file)
cmd += ' --load_average_cdf_from_file {}'.format(cdf_file)
cmd += ' --nr_of_bins {}'.format(1001)
#cmd += ' --do_not_remove_background'

print(cmd)
process = subprocess.Popen(cmd, shell=True)
process.wait()

