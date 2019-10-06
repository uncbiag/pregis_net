from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader

import glob
from utils.utils import *
import progressbar as pb
import blosc
import multiprocessing
import socket

blosc.set_nthreads(1)


class CTCBCT(Dataset):
    def __init__(self, dataset_mode='training', with_contours=True):
        root_folder = self.__setup_root_folder()
        assert (root_folder is not None)

        if with_contours:
            image_folder = os.path.join(root_folder, 'data/ct-cbct/images/with_cbct_contours')
        else:
            image_folder = os.path.join(root_folder, 'data/ct-cbct/images_without_cbct_contours')
        patients = sorted(glob.glob(os.path.join(image_folder, '18227*')))
        num_of_patients = len(patients)
        num = num_of_patients // 10
        if dataset_mode == 'training':
            self.patients = patients[0:8*num]
        elif dataset_mode == 'validation':
            self.patients = patients[8:num: 10*num]
        else:
            raise ValueError("Dataset {} not supported".format(dataset_mode))


        all_files = []

        return
        for patient in patients:
            pati




        tumor_folder = os.path.join(root_folder, 'pseudo/pseudo_3D/tumor')
        mask_folder = os.path.join(root_folder, 'pseudo/pseudo_3D/mask')

        tumor_files = sorted(glob.glob(os.path.join(tumor_folder, '*.nii.gz')))
        print(len(tumor_files))

        all_files = []
        for i in range(len(tumor_files)):
            tumor_file = tumor_files[i]
            current_file = {
                'image_name': tumor_file
            }
            tumor_name = os.path.basename(tumor_file)
            mask_name = "{}mask.nii.gz".format(tumor_name.split('tumor', 1)[0])
            mask_file = os.path.join(mask_folder, mask_name)
            if os.path.isfile(mask_file):
                current_file['mask_name'] = mask_file
            all_files.append(current_file)
        self.dataset_mode = dataset_mode
        num_of_all_files = len(all_files)
        print(num_of_all_files)
        num = num_of_all_files // 10

        if dataset_mode == 'training':
            self.files = all_files[4 * num:10 * num]
        elif dataset_mode == 'validation':
            self.files = all_files[2 * num:4 * num]
        elif dataset_mode == 'test':
            self.files = all_files[0:2 * num]
        else:
            raise ValueError('Mode not supported')

        self.atlas_file = os.path.join(root_folder, 'atlas_folder', 'atlas.nii.gz')
        image_io = py_fio.ImageIO()
        self.atlas, _, _, _ = image_io.read_to_nc_format(self.atlas_file, silent_mode=True)
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
        for files in image_list:
            image, _, _, _ = image_io.read_to_nc_format(files['image_name'], silent_mode=True)
            image_compressed = blosc.pack_array(image)
            self.files_dict[files['image_name']] = image_compressed
            if 'mask_name' in files:
                mask, _, _, _ = image_io.read_to_nc_format(files['mask_name'], silent_mode=True)
                mask_compressed = blosc.pack_array(mask)
                self.files_dict[files['mask_name']] = mask_compressed
            count += 1
            pbar.update(count)
        pbar.finish()
        return

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        global mask
        file_dict = self.files[idx]
        image_compressed = self.files_dict[file_dict['image_name']]
        image = blosc.unpack_array(image_compressed)
        if 'mask_name' in file_dict:
            mask_compressed = self.files_dict[file_dict['mask_name']]
            mask = blosc.unpack_array(mask_compressed)
            return image[0, ...], self.atlas[0, ...], mask[0, ...]
        else:
            return image[0, ...], self.atlas[0, ...]

    def __setup_root_folder(self):
        hostname = socket.gethostname()
        root_folder = None
        if hostname == 'biag-gpu0.cs.unc.edu':
            root_folder = '/playpen1/xhs400/Research/data/r21'
        elif hostname == 'biag-w05.cs.unc.edu':
            root_folder = '/playpen1/xhs400/Research/data/r21'
        return root_folder
