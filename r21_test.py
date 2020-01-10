import os
import datetime
import json
from utils.utils import *
import socket

from tensorboardX import SummaryWriter
from utils.visualize import make_image_summary
from setting import parse_opts
import pyreg.fileio as py_fio

torch.backends.cudnn.benchmark = True


class TestPregis:

    def __init__(self):
        self.dataset = 'ct_cbct'
        self.network_mode = 'mermaid'

        hostname = socket.gethostname()
        if hostname == 'biag-w05.cs.unc.edu':
            self.root_folder = '/playpen/xhs400/Research/PycharmProjects/r21_net'
        else:
            raise ValueError("Wrong host! Please configure.")
        assert (self.root_folder is not None)

        self.train_data_loader = None
        self.validate_data_loader = None
        self.pregis_net = None
        self.network_name = "model_mermaid_time_20191106-212729_initLR_0.0002_sigma_2.236_recons_5.0_seg_5.0"
        self.network_folder = os.path.join(self.root_folder, 'tmp_models', self.network_mode + '_net',
                                           self.network_name)
        self.network_file = os.path.join(self.network_folder,
                                         'best_eval.pth.tar')  # could be other validation epoch file

        self.network_config_file = None
        self.mermaid_config_file = None
        self.network_config = None
        self.mermaid_config = None
        self.__setup_configuration_file()

        self.result_folder = None
        self.__setup_output_files()

        self.__load_models()
        return

    def __setup_configuration_file(self):
        self.network_config_file = os.path.join(self.network_folder, 'network_config.json')
        self.mermaid_config_file = os.path.join(self.network_folder, 'mermaid_config.json')

        with open(self.network_config_file) as f:
            self.network_config = json.load(f)

        with open(self.mermaid_config_file) as f:
            self.mermaid_config = json.load(f)
        self.network_config['model']['mermaid_config_file'] = self.mermaid_config_file
        return

    def __setup_output_files(self):
        self.result_folder = os.path.join(self.network_folder, 'result_folder')
        os.system('mkdir -p ' + self.result_folder)
        return

    def __load_models(self):
        data_loader = create_dataloader(self.network_config, modes=['test'])
        self.test_data_loader = data_loader['test']
        self.pregis_net = create_model(self.network_config['model'], self.network_mode)
        checkpoint = torch.load(self.network_file)
        try:
            self.pregis_net.load_state_dict(checkpoint['model_state_dict'])
        except:
            print("Model load Failed")
        self.pregis_net.network_mode = self.network_mode

    def test_model(self):
        moving_paths = ['moving_image.nii.gz', 'moving_sb_label.nii.gz', 'moving_sd_label.nii.gz']
        warped_paths = ['warped_image.nii.gz', 'warped_sb_label.nii.gz', 'warped_sd_label.nii.gz']
        target_paths = ['target_image.nii.gz', 'target_sb_label.nii.gz', 'target_sd_label.nii.gz']
        map_path = 'displacement.nii.gz'

        im_io = py_fio.ImageIO()
        map_io = py_fio.MapIO()

        with torch.no_grad():
            self.pregis_net.eval()
            f = open(os.path.join(self.result_folder, 'dice.txt'), "a")
            f.write(
                "{},{},{},{},{}\n".format("case", "dice_sm_before", "dice_sd_before", "dice_sm_after", "dice_sd_after"))
            for i, image_dict in enumerate(self.test_data_loader):
                case_result_folder = os.path.join(self.result_folder, str(i + 1))
                os.system('mkdir -p ' + case_result_folder)
                weighted_mask = image_dict['weighted_mask']
                im_io.write(os.path.join(case_result_folder, 'weighted_mask.nii.gz'), torch.squeeze(weighted_mask),
                            hdr=self.network_config['model']['target_hdrc'])

                self.pregis_net(image_dict)
                loss_dict = self.pregis_net.loss_dict
                f.write("{},{},{},{},{}\n".format(str(i + 1), loss_dict['dice_SmLabel_before'],
                                                  loss_dict['dice_SdLabel_before'],
                                                  loss_dict['dice_SmLabel'], loss_dict['dice_SdLabel']))

                moving_image = image_dict['CT_image']
                moving_label1 = image_dict['CT_SmLabel']
                moving_label2 = image_dict['CT_SdLabel']
                target_image = image_dict['CBCT_image']
                target_label1 = None
                target_label2 = None

                if 'CBCT_SmLabel' in image_dict and 'CBCT_SdLabel' in image_dict:
                    target_label1 = image_dict['CBCT_SmLabel']
                    target_label2 = image_dict['CBCT_SdLabel']
                target_image_and_label = target_image
                moving_image_and_label = moving_image
                if target_label1 is not None:
                    moving_image_and_label = torch.cat((moving_image_and_label, moving_label1), dim=1)
                    target_image_and_label = torch.cat((target_image_and_label, target_label1), dim=1)
                if target_label2 is not None:
                    moving_image_and_label = torch.cat((moving_image_and_label, moving_label2), dim=1)
                    target_image_and_label = torch.cat((target_image_and_label, target_label2), dim=1)

                moving_image_and_label = moving_image_and_label
                target_image_and_label = target_image_and_label
                warped_image = self.pregis_net.warped_image.detach()
                warped_labels = self.pregis_net.warped_labels.detach()
                warped_image_and_label = torch.cat((warped_image, warped_labels), dim=1)

                for j in range(moving_image_and_label.shape[1]):
                    result_path = moving_paths[j]
                    image_to_write = moving_image_and_label[:, j, :, :, :]
                    im_io.write(os.path.join(case_result_folder, result_path), torch.squeeze(image_to_write),
                                hdr=self.network_config['model']['target_hdrc'])

                for j in range(target_image_and_label.shape[1]):
                    result_path = target_paths[j]
                    image_to_write = target_image_and_label[:, j, :, :, :]
                    im_io.write(os.path.join(case_result_folder, result_path), torch.squeeze(image_to_write),
                                hdr=self.network_config['model']['target_hdrc'])

                for j in range(warped_image_and_label.shape[1]):
                    result_path = warped_paths[j]
                    image_to_write = warped_image_and_label[:, j, :, :, :]
                    im_io.write(os.path.join(case_result_folder, result_path), torch.squeeze(image_to_write),
                                hdr=self.network_config['model']['target_hdrc'])

                # phi = self.pregis_net.phi.detach()
                # displacement = phi[0, ...] - self.pregis_net.identityMap[0, ...]
                # map_io.write(filename=os.path.join(case_result_folder, map_path), data=torch.squeeze(displacement),
                # hdr=self.network_config['model']['target_hdrc'])
            print("closing file")
            f.close()


class TrainPregis:

    def __init__(self):
        self.dataset = 'ct_cbct'

        self.settings = parse_opts()
        torch.manual_seed(self.settings.manual_seed)

        self.time = None
        # specify continue training or new training
        self.is_continue = False
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
        self.log_folder = None
        self.__setup__()

        # load models
        self.train_data_loader = None
        self.validate_data_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.__load_models__()
        print("Finish Loading models")
        return

    def __load_models__(self):
        self.train_data_loader, self.validate_data_loader = create_dataloader(self.network_config, self.settings)
        self.model = create_model(self.network_config['model'])
        self.optimizer, self.scheduler = create_optimizer(self.network_config['train'], self.model)

    def __setup__(self):
        if self.is_continue:
            # to continue, specify the model folder and model
            self.network_folder = os.path.join(os.path.dirname(__file__),
                                               "tmp_models/{}".format(self.settings.network_name))
            self.log_folder = os.path.join(os.path.dirname(__file__),
                                           "logs/{}".format(self.settings.network_name))
            self.network_config_file = os.path.join(self.network_folder, 'network_config.json')
            self.mermaid_config_file = os.path.join(self.network_folder, 'mermaid_config.json')
            self.network_file = os.path.join(self.network_folder, self.settings.saved_model)
        else:
            self.network_config_file = os.path.join(os.path.dirname(__file__),
                                                    "settings/{}/network_config.json".format(self.dataset))
            self.mermaid_config_file = os.path.join(os.path.dirname(__file__),
                                                    "settings/{}/mermaid_config.json".format(self.dataset))
            # Setup output locations, names, etc.
            now = datetime.datetime.now()
            # distinct time stamp for each model
            time = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                                  now.second)
            my_name = "model_{}".format(time)
            self.network_folder = os.path.join(os.path.dirname(__file__), 'tmp_models', my_name)
            os.system('mkdir -p ' + self.network_folder)

            print("Writing {} to {}".format(self.network_config_file,
                                            os.path.join(self.network_folder, 'network_config.json')))
            os.system('cp ' + self.network_config_file + ' ' + os.path.join(self.network_folder, 'network_config.json'))
            print("Writing {} to {}".format(self.mermaid_config_file,
                                            os.path.join(self.network_folder, 'mermaid_config.json')))
            os.system('cp ' + self.mermaid_config_file + ' ' + os.path.join(self.network_folder, 'mermaid_config.json'))
            self.log_folder = os.path.join(os.path.dirname(__file__), 'logs', my_name)
            os.system('mkdir -p ' + self.log_folder)
        with open(self.network_config_file) as f:
            self.network_config = json.load(f)

        with open(self.mermaid_config_file) as f:
            self.mermaid_config = json.load(f)
        self.network_config['model']['mermaid_config_file'] = self.mermaid_config_file
        return

    def train_model(self):
        n_epochs = self.network_config['train']['n_of_epochs']
        current_epoch = 0
        batch_size = self.network_config['train']['batch_size']
        iters_per_epoch = len(self.train_data_loader.dataset) // batch_size
        val_iters_per_epoch = len(self.validate_data_loader.dataset) // batch_size
        summary_batch_period = min(self.network_config['train']['min_summary_period'], iters_per_epoch)
        validate_epoch_period = self.network_config['validate']['validate_epoch_period']

        print('batch_size:', str(batch_size))
        print('iters_per_epoch:', str(iters_per_epoch))
        print('val_iters_per_epoch:', str(val_iters_per_epoch))
        print('summary_batch_period:', str(summary_batch_period))
        print("validate_epoch_period:", str(validate_epoch_period))

        writer = SummaryWriter(self.log_folder)
        min_val_loss = 0.0

        if self.is_continue:
            # resume training
            assert self.network_file is not None
            print("Loading previous model {}".format(self.network_file))
            checkpoint = torch.load(self.network_file)
            if 'min_val_loss' in checkpoint:
                min_val_loss = checkpoint['min_val_loss']
            if 'epoch' in checkpoint:
                current_epoch = checkpoint['epoch'] + 1
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except:
                print("Model load FAILED!!!!")

        print("Current epoch: {}".format(current_epoch))
        while current_epoch < n_epochs:

            epoch_loss_dict = {
                'all_loss': 0.0,
                'mermaid_all_loss': 0.0,
                'mermaid_reg_loss': 0.0,
                'mermaid_sim_loss': 0.0,
                'dice_SmLabel': 0.,
                'dice_SdLabel': 0.,
            }

            # self.scheduler.step(epoch=current_epoch + 1)
            iters = len(self.train_data_loader)
            self.model.train()
            for i, images in enumerate(self.train_data_loader, 0):
                self.scheduler.step(current_epoch + i / iters)
                self.optimizer.zero_grad()

                global_step = current_epoch * iters_per_epoch + (i + 1)

                ct_image = images[0].cuda()
                cb_image = images[1].cuda()
                roi_label = images[2].cuda()
                ct_sblabel = images[3].cuda()
                ct_sdlabel = images[4].cuda()
                cb_sblabel = images[5].cuda()
                cb_sdlabel = images[6].cuda()
                self.model(ct_image, cb_image, roi_label, ct_sblabel, ct_sdlabel, cb_sblabel, cb_sdlabel)

                loss_dict = self.model.loss_dict
                loss_dict['mermaid_all_loss'].backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                self.optimizer.step()

                for loss_key in epoch_loss_dict:
                    if loss_key in loss_dict:
                        epoch_loss_dict[loss_key] += loss_dict[loss_key].item()

                if (i + 1) % summary_batch_period == 0:  # print summary every k batches
                    to_print = "====>{:0d}, {:0d}, lr:{:.8f}, all_loss:{:.6f}".format(current_epoch + 1, global_step,
                                                                                      self.optimizer.param_groups[0][
                                                                                          'lr'],
                                                                                      epoch_loss_dict[
                                                                                          'all_loss'] / summary_batch_period)

                    for loss_key in epoch_loss_dict:
                        writer.add_scalar('training/training_{}'.format(loss_key),
                                          epoch_loss_dict[loss_key] / summary_batch_period, global_step=global_step)

                    to_print = to_print + ", mermaid_loss:{:.6f}, sim_loss:{:.6f}, reg_loss:{:.6f}, dice_sm:{:.6f}, " \
                                          "dice_sd:{:.6f}".format(
                        epoch_loss_dict['mermaid_all_loss'] / summary_batch_period,
                        epoch_loss_dict['mermaid_sim_loss'] / summary_batch_period,
                        epoch_loss_dict['mermaid_reg_loss'] / summary_batch_period,
                        epoch_loss_dict['dice_SmLabel'] / summary_batch_period,
                        epoch_loss_dict['dice_SdLabel'] / summary_batch_period
                    )

                    images_to_show = {
                        "ct_image": ct_image.cpu(),
                        "cb_image": cb_image.cpu(),
                        "warped_image": self.model.warped_image.detach().cpu()
                    }
                    labels_to_show = {
                        "ct_sblabel": ct_sblabel.cpu(),
                        "ct_sdlabel": ct_sdlabel.cpu(),
                        "cb_sblabel": cb_sblabel.cpu(),
                        "cb_sdlabel": cb_sdlabel.cpu(),
                        "warped_sblabel": self.model.warped_labels.detach().cpu()[:, [0], ...],
                        "warped_sdlabel": self.model.warped_labels.detach().cpu()[:, [1], ...],
                        "roi_label": roi_label.cpu()
                    }

                    phis_to_show = [self.model.phi.detach().cpu()]

                    image_summary = make_image_summary(images_to_show, labels_to_show, phis_to_show)
                    for key, value in image_summary.items():
                        writer.add_image("training_" + key, value, global_step=global_step)

                    print(to_print)
                    epoch_loss_dict = {
                        'all_loss': 0.0,
                        'mermaid_all_loss': 0.0,
                        'mermaid_reg_loss': 0.0,
                        'mermaid_sim_loss': 0.0,
                        'dice_SmLabel': 0.,
                        'dice_SdLabel': 0.,
                    }
                    writer.flush()
            if current_epoch % validate_epoch_period == 0:  # validate every k epochs
                eval_loss_dict = {
                    'all_loss': 0.0,
                    'mermaid_all_loss': 0.0,
                    'mermaid_reg_loss': 0.0,
                    'mermaid_sim_loss': 0.0,
                    'dice_SmLabel': 0.,
                    'dice_SdLabel': 0.,
                }
                self.model.eval()

                with torch.no_grad():
                    validate_index = np.random.randint(0, len(self.validate_data_loader))
                    for j, images in enumerate(self.validate_data_loader, 0):
                        ct_image = images[0].cuda()
                        cb_image = images[1].cuda()
                        roi_label = images[2].cuda()
                        ct_sblabel = images[3].cuda()
                        ct_sdlabel = images[4].cuda()
                        cb_sblabel = images[5].cuda()
                        cb_sdlabel = images[6].cuda()
                        self.model(ct_image, cb_image, roi_label, ct_sblabel, ct_sdlabel, cb_sblabel, cb_sdlabel)

                        loss_dict = self.model.loss_dict
                        for loss_key in eval_loss_dict:
                            if loss_key in loss_dict:
                                eval_loss_dict[loss_key] += loss_dict[loss_key].item()

                        if j == validate_index:
                            # view validation result
                            images_to_show = {
                                "ct_image": ct_image.cpu(),
                                "cb_image": cb_image.cpu(),
                                "warped_image": self.model.warped_image.detach().cpu()
                            }
                            labels_to_show = {
                                "ct_sblabel": ct_sblabel.cpu(),
                                "ct_sdlabel": ct_sdlabel.cpu(),
                                "cb_sblabel": cb_sblabel.cpu(),
                                "cb_sdlabel": cb_sdlabel.cpu(),
                                "warped_sblabel": self.model.warped_labels.detach().cpu()[:, [0], ...],
                                "warped_sdlabel": self.model.warped_labels.detach().cpu()[:, [1], ...],
                                "roi_label": roi_label.cpu()
                            }

                            phis_to_show = [self.model.phi.detach().cpu()]

                            image_summary = make_image_summary(images_to_show, labels_to_show, phis_to_show)
                            for key, value in image_summary.items():
                                writer.add_image("validation_" + key, value, global_step=current_epoch)

                for loss_key in eval_loss_dict:
                    writer.add_scalar('validation/validation_{}'.format(loss_key),
                                      eval_loss_dict[loss_key] / val_iters_per_epoch, global_step=current_epoch)

                to_print = "EVAL>{:0d}, all_loss:{:.6f}".format(current_epoch + 1,
                                                                eval_loss_dict['all_loss'] / val_iters_per_epoch)
                to_print = to_print + ", mermaid_loss:{:.6f}, sim_loss:{:.6f}, reg_loss:{:.6f}, dice_sm:{:.6f}, dice_sd:{:.6f}".format(
                    eval_loss_dict['mermaid_all_loss'] / val_iters_per_epoch,
                    eval_loss_dict['mermaid_sim_loss'] / val_iters_per_epoch,
                    eval_loss_dict['mermaid_reg_loss'] / val_iters_per_epoch,
                    eval_loss_dict['dice_SmLabel'] / val_iters_per_epoch,
                    eval_loss_dict['dice_SdLabel'] / val_iters_per_epoch
                )
                print(to_print)
                if min_val_loss == 0.0 and current_epoch >= 20:
                    min_val_loss = eval_loss_dict['mermaid_all_loss']
                if eval_loss_dict['mermaid_all_loss'] < min_val_loss:
                    min_val_loss = eval_loss_dict['mermaid_all_loss']
                    save_file = os.path.join(self.network_folder, 'best_eval.pth.tar')
                    print("Writing current best eval model")
                    torch.save({'epoch': current_epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict()},
                               save_file)

                if current_epoch % 20 == 0:
                    save_file = os.path.join(self.network_folder, 'eval_' + str(current_epoch) + '.pth.tar')
                    torch.save({'epoch': current_epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict()},
                               save_file)
                writer.flush()
            current_epoch = current_epoch + 1


if __name__ == '__main__':
    # network = TestPregis()
    # network.test_model()
    network = TrainPregis()
    network.train_model()
