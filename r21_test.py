import os
import sys
import datetime
import json
import socket

from setting import parse_opts
import torch
import torch.nn.functional as F
import SimpleITK as sitk

torch.backends.cudnn.benchmark = True
sys.path.append(os.path.join(os.path.dirname(__file__), '../mermaid'))
from modules.mermaid_net import MermaidNet
from torch.utils.data import DataLoader
from data_loaders.ct_cbct import R21RegDataset
import numpy as np


class TestR21:
    def __init__(self):
        self.dataset = 'ct_cbct'
        self.settings = parse_opts()
        torch.manual_seed(self.settings.manual_seed)

        self.time = None
        # specify continue training or new training
        self.is_continue = True
        self.root_folder = None

        hostname = socket.gethostname()
        if hostname == 'biag-w05.cs.unc.edu':
            self.root_folder = '/playpen/xhs400/Research/PycharmProjects/r21_net'
        else:
            raise ValueError("Wrong host! Please configure.")
        assert (self.root_folder is not None)

        # set configuration file:
        self.network_config = None
        self.mermaid_config = None
        self.network_folder = None
        self.network_file = None
        self.network_config_file = None
        self.mermaid_config_file = None
        self.img_list = None
        self.log_folder = None
        self.test_folder = None
        self.__setup__()

        # load models
        self.test_data_loader = None
        self.model = None
        self.__load_models__()
        print("Finish Loading models")
        return

    def __create_test_model__(self):
        model = MermaidNet(self.network_config['model'])
        model.cuda()
        checkpoint = torch.load(self.network_file)
        print("Best eval epoch: {}".format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def __create_test_dataloader__(self):
        model_config = self.network_config['model']
        test_dataset = R21RegDataset(self.settings, 'test')

        batch_size = 1
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     drop_last=False, num_workers=4)
        # add info to config
        model_config['img_sz'] = [batch_size, 1, self.settings.input_D, self.settings.input_H, self.settings.input_W]
        model_config['dim'] = 3
        return test_dataloader

    def __load_models__(self):
        self.test_data_loader = self.__create_test_dataloader__()
        self.model = self.__create_test_model__()

    def __setup__(self):
        # to continue, specify the model folder and model
        self.network_folder = os.path.join(os.path.dirname(__file__),
                                           "tmp_models/{}".format(self.settings.network_name))
        self.log_folder = os.path.join(os.path.dirname(__file__),
                                       "logs/{}".format(self.settings.network_name))
        self.network_config_file = os.path.join(self.network_folder, 'network_config.json')
        self.mermaid_config_file = os.path.join(self.network_folder, 'mermaid_config.json')
        self.network_file = os.path.join(self.network_folder, self.settings.saved_model)

        self.test_folder = os.path.join(self.network_folder, "test_result_{}".format(self.settings.saved_model))
        os.system('mkdir {}'.format(self.test_folder))

        print("Loading {}".format(self.network_config_file))
        print("Loading {}".format(self.mermaid_config_file))
        with open(self.network_config_file) as f:
            self.network_config = json.load(f)
        with open(self.mermaid_config_file) as f:
            self.mermaid_config = json.load(f)
        self.network_config['model']['mermaid_config_file'] = self.mermaid_config_file

        #print("Reading {}".format(self.settings.test_list))
        with open(self.settings.test_list, 'r') as f:
            self.img_list = [line.strip() for line in f]
        return

    def __calculate_dice_score__(self, predicted_label, target_label, mask=None):
        if mask is not None:
            roi_indices = np.where(mask > 0.5)
            predicted_in_mask = predicted_label[roi_indices]
            target_in_mask = target_label[roi_indices]
        else:
            predicted_in_mask = predicted_label
            target_in_mask = target_label
        intersection = (predicted_in_mask * target_in_mask).sum()
        smooth = 1.
        dice = (2 * intersection + smooth) / (predicted_in_mask.sum() + target_in_mask.sum() + smooth)
        return dice

    def test_model(self):
        iters = len(self.test_data_loader.dataset)
        print('iters:', str(iters))

        mergeFilter = sitk.MergeLabelMapFilter()
        mergeFilter.SetMethod(Method=0)
        dice_file = os.path.join(self.test_folder, 'dice.txt')
        dice_all_file = os.path.join(self.test_folder, 'dice_all.txt')
        f = open(dice_file, 'w')
        # f2 = open(dice_all_file, 'w')
        with torch.no_grad():
            for j, images in enumerate(self.test_data_loader, 0):
                ct_image_name = self.img_list[j].split(' ')[0]
                cb_image_name = self.img_list[j].split(' ')[1]

                if not ("OG" in ct_image_name and "OG" in cb_image_name):
                    continue
                patient = ct_image_name.split('18227')[1][0:2]
                cb_case = cb_image_name.split('CBCT')[1][0:2]
                ct_image = images[0].cuda()
                cb_image = images[1].cuda()
                roi_label = images[2].cuda()
                ct_sblabel = images[3].cuda()
                ct_sdlabel = images[4].cuda()
                cb_sblabel = images[5].cuda()
                cb_sdlabel = images[6].cuda()
                self.model(ct_image, cb_image, roi_label, ct_sblabel, ct_sdlabel, cb_sblabel, cb_sdlabel)
                warped_image = self.model.warped_image
                warped_sblabel = self.model.warped_labels[:, [0], ...]
                warped_sdlabel = self.model.warped_labels[:, [1], ...]

                # phi = self.model.phi - self.model.identityMap
                # for dim in range(3):
                #     phi[0, dim, ...] = phi[0, dim, ...] / self.model.spacing[dim]

                result_folder = os.path.join(self.test_folder,
                                             '{}__to_{}'.format(ct_image_name.split('images/')[1].replace('/', '_'),
                                                                cb_image_name.split('images')[1].replace('/', '_')))
                # print("Result folder: {}".format(re09876sult_folder))
                os.system('mkdir -p {}'.format(result_folder))
                orig_image_itk = sitk.ReadImage(ct_image_name)

                orig_image_arr = sitk.GetArrayFromImage(orig_image_itk)
                [depth, height, width] = orig_image_arr.shape
                # print("Original image shape: {}".format(orig_image_arr.shape))
                scale = [depth * 1.0 / self.settings.input_D,
                         height * 1.0 / self.settings.input_H * 1.0,
                         width * 1.0 / self.settings.input_W * 1.0]

                orig_warped_image = F.interpolate(warped_image, scale_factor=scale, mode='trilinear')
                orig_warped_sblabel = F.interpolate(warped_sblabel, scale_factor=scale, mode='nearest')
                orig_warped_sdlabel = F.interpolate(warped_sdlabel, scale_factor=scale, mode='nearest')
                # orig_phi_x = F.interpolate(phi[:, [0], ...], scale_factor=scale, mode='trilinear')
                # orig_phi_y = F.interpolate(phi[:, [1], ...], scale_factor=scale, mode='trilinear')
                # orig_phi_z = F.interpolate(phi[:, [2], ...], scale_factor=scale, mode='trilinear')

                orig_warped_image_itk = sitk.GetImageFromArray(torch.squeeze(orig_warped_image).cpu().numpy())
                orig_warped_image_itk.CopyInformation(orig_image_itk)
                orig_warped_image_file = os.path.join(result_folder, 'warped_image.nii.gz')
                sitk.WriteImage(orig_warped_image_itk, orig_warped_image_file)

                orig_warped_sblabel_arr = torch.squeeze(orig_warped_sblabel).cpu().numpy().astype(np.uint8)
                orig_warped_sblabel_itk = sitk.GetImageFromArray(orig_warped_sblabel_arr)
                orig_warped_sblabel_itk.CopyInformation(orig_image_itk)
                orig_warped_sblabel_file = os.path.join(result_folder, 'warped_sblabel.nii.gz')
                sitk.WriteImage(orig_warped_sblabel_itk, orig_warped_sblabel_file)

                orig_warped_sdlabel_arr = torch.squeeze(orig_warped_sdlabel).cpu().numpy().astype(np.uint8)
                orig_warped_sdlabel_itk = sitk.GetImageFromArray(orig_warped_sdlabel_arr)
                orig_warped_sdlabel_itk.CopyInformation(orig_image_itk)
                orig_warped_sdlabel_file = os.path.join(result_folder, 'warped_sdlabel.nii.gz')
                sitk.WriteImage(orig_warped_sdlabel_itk, orig_warped_sdlabel_file)

                all_labels_itk = mergeFilter.Execute([sitk.Cast(orig_warped_sblabel_itk, sitk.sitkLabelUInt8),
                                                      sitk.Cast(orig_warped_sdlabel_itk, sitk.sitkLabelUInt8)])
                all_labels_file = os.path.join(result_folder, 'warped_labels.nii.gz')
                sitk.WriteImage(sitk.Cast(all_labels_itk, sitk.sitkUInt8), all_labels_file)

                # orig_phi = torch.cat((orig_phi_x, orig_phi_y, orig_phi_z), dim=1).permute([0, 2, 3, 4, 1])
                # print("Transformation map shape: {}".format(orig_phi.shape))
                # orig_phi_itk = sitk.GetImageFromArray(torch.squeeze(orig_phi).cpu().numpy(), isVector=True)
                # orig_phi_itk.CopyInformation(orig_image_itk)
                # orig_phi_file = os.path.join(result_folder, 'phi.nii')
                # sitk.WriteImage(orig_phi_itk, orig_phi_file)

                ct_sblabel = self.img_list[j].split(' ')[3]
                ct_sblabel_arr = sitk.GetArrayFromImage(sitk.ReadImage(ct_sblabel))
                ct_sdlabel = self.img_list[j].split(' ')[4]
                ct_sdlabel_arr = sitk.GetArrayFromImage(sitk.ReadImage(ct_sdlabel))

                cb_sblabel = self.img_list[j].split(' ')[5]
                cb_sblabel_arr = sitk.GetArrayFromImage(sitk.ReadImage(cb_sblabel))
                cb_sdlabel = self.img_list[j].split(' ')[6]
                cb_sdlabel_arr = sitk.GetArrayFromImage(sitk.ReadImage(cb_sdlabel))

                roi_label = self.img_list[j].split(' ')[2]
                roi_arr = sitk.GetArrayFromImage(sitk.ReadImage(roi_label))

                sm_label_bef = self.__calculate_dice_score__(cb_sblabel_arr, ct_sblabel_arr, roi_arr)
                sd_label_bef = self.__calculate_dice_score__(cb_sdlabel_arr, ct_sdlabel_arr, roi_arr)
                # sm_label_dice = self.__calculate_dice_score__(torch.squeeze(warped_sblabel).cpu().numpy(),
                #                                               torch.squeeze(images[3]).numpy(),
                #                                               torch.squeeze(images[2]).numpy())
                # sd_label_dice = self.__calculate_dice_score__(torch.squeeze(warped_sdlabel).cpu().numpy(),
                #                                               torch.squeeze(images[4]).numpy(),
                #                                               torch.squeeze(images[2]).numpy())
                sm_label_dice = self.__calculate_dice_score__(orig_warped_sblabel_arr, ct_sblabel_arr, roi_arr)
                sd_label_dice = self.__calculate_dice_score__(orig_warped_sdlabel_arr, ct_sdlabel_arr, roi_arr)
                print('{}, {}, {}, {}, {}, {}'.format(patient, cb_case, sm_label_bef,
                                                      sd_label_bef,
                                                      sm_label_dice,
                                                      sd_label_dice))
                f.write('{}__to_{}, {}, {}, {}, {}\n'.format(ct_image_name.split('images/')[1].replace('/', '_'),
                                                             cb_image_name.split('images')[1].replace('/', '_'),
                                                             sm_label_bef,
                                                             sd_label_bef,
                                                             sm_label_dice,
                                                             sd_label_dice))

                # sm_label_bef = self.__calculate_dice_score__(cb_sblabel_arr, ct_sblabel_arr)
                # sd_label_bef = self.__calculate_dice_score__(cb_sdlabel_arr, ct_sdlabel_arr)
                # # sm_label_dice = self.__calculate_dice_score__(torch.squeeze(warped_sblabel).cpu().numpy(),
                # #                                               torch.squeeze(images[3]).numpy(),
                # #                                               torch.squeeze(images[2]).numpy())
                # # sd_label_dice = self.__calculate_dice_score__(torch.squeeze(warped_sdlabel).cpu().numpy(),
                # #                                               torch.squeeze(images[4]).numpy(),
                # #                                               torch.squeeze(images[2]).numpy())
                # sm_label_dice = self.__calculate_dice_score__(orig_warped_sblabel_arr, ct_sblabel_arr)
                # sd_label_dice = self.__calculate_dice_score__(orig_warped_sdlabel_arr, ct_sdlabel_arr)
                # # print('{}, {}, {}, {}'.format(sm_label_bef,
                # #                               sd_label_bef,
                # #                               sm_label_dice,
                # #                               sd_label_dice))
                # f2.write('{}__to_{}, {}, {}, {}, {}\n'.format(ct_image_name.split('images/')[1].replace('/', '_'),
                #                                               cb_image_name.split('images')[1].replace('/', '_'),
                #                                               sm_label_bef,
                #                                               sd_label_bef,
                #                                               sm_label_dice,
                #                                               sd_label_dice))

        f.close()
        # f2.close()


if __name__ == '__main__':
    network = TestR21()
    network.test_model()
