from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader

import glob
from utils.utils import *
import progressbar as pb
import blosc
import multiprocessing
import socket

blosc.set_nthreads(1)


class Pseudo3DDataset(Dataset):
    def __init__(self, dataset_type, dataset_mode):
        root_folder = self.__setup_root_folder()
        assert (root_folder is not None)
        self.num_of_workers = 20

        if dataset_type == 'normal':
            # tumor_folder = os.path.join(root_folder, 'pseudo/pseudo_3D/no_tumor')
            # mask_folder = None
            raise ValueError("Temporarily disable loading all normals")
        elif dataset_type == 'tumor':
            tumor_folder = os.path.join(root_folder, 'pseudo/pseudo_3D/tumor')
            # mask_folder = os.path.join(root_folder, 'pseudo/pseudo_3D/mask')
            # disp_folder = os.path.join(root_folder, 'pseudo/pseudo_3D/atlas_warp_disp')
        else:
            raise ValueError("dataset type wrong")

        tumor_files = sorted(glob.glob(os.path.join(tumor_folder, '*.nii.gz')))

        all_files = []  # a list of dictionaries. Each dictionary contains image_name and mask_name
        for i in range(len(tumor_files)):
            tumor_file = tumor_files[i]
            mask_file = tumor_file.replace('tumor', 'mask')
            current_file = {
                'image_name': tumor_file,
                'mask_name': mask_file,
                'disp_name': None
            }
            if dataset_mode == 'validate' or dataset_mode == 'test':
                disp_file = tumor_file.replace('tumor', 'no_tumor_atlas_warp_disp')
                current_file['disp_name'] = disp_file
            all_files.append(current_file)
        self.dataset_mode = dataset_mode
        num_of_all_files = len(all_files)
        num = num_of_all_files // 10

        if dataset_mode == 'train':
            self.files = all_files[4 * num:10 * num]
        elif dataset_mode == 'validate':
            self.files = all_files[2 * num:4 * num]
        elif dataset_mode == 'test':
            self.files = all_files[0:2 * num]
        else:
            raise ValueError('Mode not supported')

        self.atlas_file = os.path.join(root_folder, 'atlas_folder', 'atlas.nii.gz')
        image_io = py_fio.ImageIO()
        atlas_unnorm, _, _, _ = image_io.read_to_nc_format(self.atlas_file, silent_mode=True)
        self.atlas = self.__normalize_intensity(atlas_unnorm)
        self.images_dict = None
        self.__multi_threads_loading()
        return

    def __multi_threads_loading(self):
        manager = multiprocessing.Manager()
        self.files_dict = manager.dict()
        split_list = self.__split_files()
        process = []
        for i in range(self.num_of_workers):
            p = multiprocessing.Process(target=self.__load_images_and_compress, args=(split_list[i],))
            p.start()
            process.append(p)

        for p in process:
            p.join()
        self.files_dict = dict(self.files_dict)
        print("Finish loading images")
        return

    def __split_files(self):
        index_list = list(range(len(self.files)))
        index_split = np.array_split(np.array(index_list), self.num_of_workers)
        split_list = []
        for i in range(self.num_of_workers):
            current_list = self.files[index_split[i][0]:index_split[i][0] + len(index_split[i])]
            split_list.append(current_list)
        return split_list

    def __load_images_and_compress(self, image_list):
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(image_list)).start()
        count = 0
        image_io = py_fio.ImageIO()
        map_io = py_fio.MapIO()
        for files in image_list:
            image, _, _, _ = image_io.read_to_nc_format(files['image_name'], silent_mode=True)
            image_compressed = blosc.pack_array(self.__normalize_intensity(image))
            self.files_dict[files['image_name']] = image_compressed

            mask, _, _, _ = image_io.read_to_nc_format(files['mask_name'], silent_mode=True)
            mask_compressed = blosc.pack_array(mask)
            self.files_dict[files['mask_name']] = mask_compressed

            if files['disp_name'] is not None:
                disp, _, _, _ = map_io.read_from_validation_map_format(files['disp_name'])
                disp_compressed = blosc.pack_array(disp)
                self.files_dict[files['disp_name']] = disp_compressed

            count += 1
            pbar.update(count)
        pbar.finish()
        return

    def __normalize_intensity(self, img):
        normalized_img = img * 2 - 1
        return normalized_img

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        global mask
        file_dict = self.files[idx]
        image_compressed = self.files_dict[file_dict['image_name']]
        image = blosc.unpack_array(image_compressed)
        mask_compressed = self.files_dict[file_dict['mask_name']]
        mask = blosc.unpack_array(mask_compressed)

        if file_dict['disp_name'] is not None :
            disp_compressed = self.files_dict[file_dict['disp_name']]
            disp = blosc.unpack_array(disp_compressed)
            return self.atlas[0, ...], image[0, ...], mask[0, ...], disp
        else:
            return self.atlas[0, ...], image[0, ...], mask[0, ...]

    def __setup_root_folder(self):
        hostname = socket.gethostname()
        root_folder = None
        if hostname == 'biag-gpu0.cs.unc.edu':
            root_folder = '/playpen/xhs400/Research/data/data_for_pregis_net'
        elif hostname == 'biag-w05.cs.unc.edu':
            root_folder = '/playpen1/xhs400/Research/data/data_for_pregis_net'
        elif 'lambda' in hostname:
            root_folder = '/playpen/xhs400/Research/data/data_for_pregis_net'
        return root_folder
