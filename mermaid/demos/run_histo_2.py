import os
import sys
import subprocess
import glob

folder = int(sys.argv[1])
if folder == 1:
    train_range = range(1,18)
    valid_range = range(18,21)
    test_range = range(21, 41)
    print(train_range, valid_range, test_range)
elif folder == 2:
    train_range = range(21, 38)
    valid_range = range(38,41)
    test_range = (1,21)
else:
    raise ValueError('Wrong')

mode = sys.argv[2]
assert(mode == 'train' or mode == 'test')


root_folder = '/playpen/xhs400/Research/data/penpen/LPBA'
image_folder = os.path.join(root_folder , 'folder_' + str(folder), mode) 
all_cases = os.listdir(image_folder)
brain_folder = os.path.join(root_folder, 'brain_affine_icbm_hist_oasis')
normal_image_folder = os.path.join(brain_folder, 'folder_' + str(folder) + '_training')
cdf_file = os.path.join(normal_image_folder, 'average_cdf.pt')
for case in all_cases:
    case_folder = os.path.join(image_folder, case)
    src_to_target_folder = os.listdir(case_folder)
    for src in src_to_target_folder:
        final_warped_folder = os.path.join(case_folder, src)
        warped_image = os.path.join(final_warped_folder, 'I1.nii.gz')
        matched_image = os.path.join(final_warped_folder, 'I1_hist.nii.gz')
        os.system('rm {}'.format(matched_image)) 
        matched_folder = os.path.join(final_warped_folder, 'histogram_matched')
        os.system('mkdir {}'.format(matched_folder))
        cmd = "python normalize_image_intensities.py "

        cmd += ' --desired_output_directory {} '.format(matched_folder)
        cmd += ' --image_to_normalize {} '.format(warped_image)
        cmd += ' --load_average_cdf_from_file {}'.format(cdf_file)
        cmd += ' --nr_of_bins {}'.format(1000)

        print(cmd)
        process = subprocess.Popen(cmd, shell=True)
        process.wait()

