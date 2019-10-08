from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader

import glob
from utils.utils import *
import progressbar as pb
import blosc
import multiprocessing
import socket

blosc.set_nthreads(1)


class CBCTDataset(Dataset):
    def __init__(self, dataset_type = 'with_contour', dataset_mode='training'):
        super(CBCTDataset, self).__init__()
        root_folder = self.__setup_root_folder()
        self.num_of_workers = 8
        assert (root_folder is not None)

        if dataset_type == 'with_contours':
            image_folder = os.path.join(root_folder, 'data/ct-cbct/images/with_cbct_contours')
        elif dataset_type == 'without_contours':
            image_folder = os.path.join(root_folder, 'data/ct-cbct/images_without_cbct_contours')
        else:
            raise ValueError("Dataset type wrong")
        patients = sorted(glob.glob(os.path.join(image_folder, '18227*')))
        num_of_patients = len(patients)
        num = num_of_patients // 10
        if dataset_mode == 'train':
            self.patients = patients[0:8*num]
        elif dataset_mode == 'validate':
            self.patients = patients[8*num: 10*num]
        else:
            raise ValueError("Dataset {} not supported".format(dataset_mode))

        self.all_cases = []

        for patient_folder in self.patients:
            cases_folder = glob.glob(os.path.join(patient_folder, "*cropped_images"))
            for image_folder in cases_folder:
                image_case = image_folder.split('-01_cropped_images')[0][-2:]
                file_dict = [os.path.join(image_folder, '1900-01__Studies_image_cropped_{}.nii.gz'.format(image_case)),
                             os.path.join(image_folder, '1900-01__Studies_SmBowel_label.nii.gz'),
                             os.path.join(image_folder, '1900-01__Studies_StomachDuo_label.nii.gz'),
                             os.path.join(image_folder, '1900-01__Studies_weighted_mask.nii.gz'),
                             os.path.join(image_folder, '1900-01__Studies_weighted_mask_before_smooth.nii.gz'),
                             os.path.join(image_folder, '19{}-01__Studies_image_normalized.nii.gz'.format(image_case)),
                             os.path.join(image_folder, '19{}-01__Studies_SmBowel_label.nii.gz'.format(image_case)),
                             os.path.join(image_folder, '19{}-01__Studies_StomachDuo_label.nii.gz'.format(image_case))]
                self.all_cases.append(file_dict)

        self.__multi_threads_loading()
        image_io = py_fio.ImageIO()
        _, target_hdrc, target_spacing, _ = image_io.read_to_nc_format(self.all_cases[0][0], silent_mode=True)
        self.target_hdrc = target_hdrc
        self.target_spacing = target_spacing

        self.sz = [192, 192, 192]  # fixed patch size
        return

    def __multi_threads_loading(self):
        manager = multiprocessing.Manager()
        self.files_dict = manager.dict()
        self.roi_dict = manager.dict()
        split_list = self.__split_files()
        process = []
        for i in range(self.num_of_workers):
            p = multiprocessing.Process(target=self.__load_images_and_compress, args=(split_list[i],))
            p.start()
            process.append(p)

        for p in process:
            p.join()
        self.files_dict = dict(self.files_dict)
        self.roi_dict = dict(self.roi_dict)
        print("Finish loading images")
        return

    def __split_files(self):
        index_list = list(range(len(self.all_cases)))
        index_split = np.array_split(np.array(index_list), self.num_of_workers)
        split_list = []
        for i in range(self.num_of_workers):
            current_list = self.all_cases[index_split[i][0]:index_split[i][0] + len(index_split[i])]
            split_list.append(current_list)
        return split_list


    def __get_mid_roi(self, image):
        r = np.any(image, axis=(1, 2))
        c = np.any(image, axis=(0, 2))
        z = np.any(image, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        mid_point = np.array([rmin + (rmax - rmin)//2, cmin + (cmax - cmin)//2, zmin + (zmax - zmin)//2])
        return mid_point

    def __load_images_and_compress(self, image_list):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(image_list) * len(image_list[0])).start()
        count = 0
        image_io = py_fio.ImageIO()
        for patient_cases in image_list:
            for file_name in patient_cases:
                if os.path.isfile(file_name):
                    image, _, _, _ = image_io.read_to_nc_format(file_name, silent_mode=True)
                    image_compressed = blosc.pack_array(image)
                    if os.path.basename(file_name) == '1900-01__Studies_weighted_mask_before_smooth.nii.gz':
                        mid_point = self.__get_mid_roi(image[0, 0, ...])
                        self.roi_dict[file_name] = mid_point
                    else:
                        self.files_dict[file_name] = image_compressed

                count += 1
                pbar.update(count)
        pbar.finish()
        return

    def __len__(self):
        return len(self.all_cases)

    def __getitem__(self, idx):
        file_dict = self.all_cases[idx]
        weighted_mask_before_smooth_name = file_dict[4]
        roi = self.roi_dict[weighted_mask_before_smooth_name]
        random_shift = np.random.randint(-10, 21, size=(3,))
        new_center = roi + random_shift
        output_dict = {}
        for file_name in file_dict:
            image_case = file_name.split('-01_cropped_images')[0][-2:]
            if os.path.isfile(file_name):
                if os.path.basename(file_name) == "1900-01__Studies_weighted_mask_before_smooth.nii.gz":
                    continue
                image_compressed = self.files_dict[file_name]
                image_full = blosc.unpack_array(image_compressed)
                image = image_full[0, [0],
                        max(min(new_center[0]-96, 32), 0):max(min(new_center[0]+96, 224), 192),
                        (new_center[1]-96):(new_center[1]+96),
                        (new_center[2]-96):(new_center[2]+96)]
                if os.path.basename(file_name) == "1900-01__Studies_image_cropped_{}.nii.gz".format(image_case):
                    output_dict['CT_image'] = image
                elif os.path.basename(file_name) == "1900-01__Studies_SmBowel_label.nii.gz":
                    output_dict['CT_SmLabel'] = image
                elif os.path.basename(file_name) == "1900-01__Studies_StomachDuo_label.nii.gz":
                    output_dict['CT_SdLabel'] = image
                elif os.path.basename(file_name) == "1900-01__Studies_weighted_mask.nii.gz":
                    output_dict['weighted_mask'] = image
                elif os.path.basename(file_name) == "19{}-01__Studies_image_normalized.nii.gz".format(image_case):
                    output_dict['CBCT_image'] = image
                elif os.path.basename(file_name) == "19{}-01__Studies_SmBowel_label.nii.gz".format(image_case):
                    output_dict['CBCT_SmLabel'] = image
                elif os.path.basename(file_name) == "19{}-01__Studies_StomachDuo_label.nii.gz".format(image_case):
                    output_dict['CBCT_SdLabel'] = image
                elif os.path.basename(file_name) == "1900-01__Studies_weighted_mask_before_smooth.nii.gz":
                    output_dict['weighted_mask_bs'] = image
                else:
                    raise ValueError("Wrong file name, check")
        return output_dict


    def __setup_root_folder(self):
        hostname = socket.gethostname()
        root_folder = None
        if hostname == 'biag-gpu0.cs.unc.edu':
            root_folder = '/playpen1/xhs400/Research/data/r21'
        elif hostname == 'biag-w05.cs.unc.edu':
            root_folder = '/playpen1/xhs400/Research/data/r21'
        return root_folder
