import os
import datetime
import json
from utils.utils import *
import socket

from tensorboardX import SummaryWriter
from utils.visualize import make_image_summary


class TrainPregis:

    def __init__(self):
        self.dataset = 'pseudo_3D'
        # network_mode selected from  'mermaid', 'recons', 'pregis'
        self.network_mode = 'pregis'

        self.time = None
        # specify continue training or new training
        self.is_continue = False
        #
        self.root_folder = None
        hostname = socket.gethostname()
        if hostname == 'biag-gpu0.cs.unc.edu':
            self.root_folder = '/playpen/xhs400/Research/pregis_net'
        elif hostname == 'biag-w05.cs.unc.edu':
            self.root_folder = '/playpen/xhs400/Research/PycharmProjects/pregis_net'
        else:
            raise ValueError("Wrong host! Please configure.")
        assert (self.root_folder is not None)

        # set configuration file:
        self.network_config = None
        self.mermaid_config = None
        self.network_folder = None
        self.network_file = {}
        self.network_config_file = None
        self.mermaid_config_file = None
        self.__setup_configuration_file()

        # load models
        self.train_data_loader = None
        self.validate_data_loader = None
        self.tumor_data_loader = None
        self.pregis_net = None
        self.optimizer = None
        self.scheduler = None
        self.__load_models()

        self.log_folder = None
        self.__setup_output_files()

    def __setup_configuration_file(self):
        if self.is_continue:
            # to continue, specify the model folder and model
            if self.network_mode == 'pregis':
                self.network_folder = ""
                self.network_file['pregis_net'] = os.path.join(self.network_folder, "")
                self.network_config_file = os.path.join(self.network_folder, 'pregis_network_config.json')

            self.mermaid_config_file = os.path.join(self.network_folder, 'mermaid_config.json')
        else:
            self.network_config_file = os.path.join(os.path.dirname(__file__),
                                                    "settings/{}/network_config.json".format(self.dataset))

            self.mermaid_config_file = os.path.join(os.path.dirname(__file__),
                                                    "settings/{}/mermaid_config.json".format(self.dataset))

        with open(self.network_config_file) as f:
            self.network_config = json.load(f)

        with open(self.mermaid_config_file) as f:
            self.mermaid_config = json.load(f)
        self.network_config['model']['mermaid_config_file'] = self.mermaid_config_file

    def __load_models(self):
        train_config = self.network_config['train']

        self.train_data_loader, self.validate_data_loader, self.tumor_data_loader = \
            create_dataloader(self.network_config['model'], train_config)
        self.pregis_net = create_model(self.network_config['model'], self.network_mode)
        self.pregis_net.network_mode = self.network_mode

        if self.network_mode == 'pregis':
            self.optimizer, self.scheduler = create_optimizer(train_config, self.pregis_net)
        else:
            raise ValueError("Wrong network mode")

    def __setup_output_files(self):
        # Setup output locations, names, etc.
        now = datetime.datetime.now()
        # distinct time stamp for each model
        time = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(now.year,
                                                              now.month,
                                                              now.day,
                                                              now.hour,
                                                              now.minute,
                                                              now.second)
        init_lr = self.network_config['train']['optimizer']['lr']
        my_name = "model_{}_time_{}_initLR_{}".format(self.network_mode, time, init_lr)
        sigma = self.mermaid_config['model']['registration_model']['similarity_measure']['sigma']
        my_name = my_name + '_sigma_{}'.format(sigma)

        seg_weight = self.network_config['model']['pregis_net']['seg_weight']
        recons_weight = self.network_config['model']['pregis_net']['recons_weight']
        my_name = my_name + '_recons_{}_seg_{}'.format(recons_weight, seg_weight)

        self.network_folder = os.path.join(os.path.dirname(__file__),
                                           'tmp_models',
                                           '{}_net'.format(self.network_mode),
                                           my_name)
        os.system('mkdir -p ' + self.network_folder)
        if self.network_mode == 'mermaid':
            print("Writing {} to {}".format(self.network_config_file,
                                            os.path.join(self.network_folder, 'mermaid_network_config.json')))
            os.system('cp ' + self.network_config_file + ' ' + os.path.join(self.network_folder,
                                                                            'recons_network_config.json'))
        if self.network_mode == 'recons':
            print("Writing {} to {}".format(self.network_config_file,
                                            os.path.join(self.network_folder, 'recons_network_config.json')))
            os.system('cp ' + self.network_config_file + ' ' + os.path.join(self.network_folder,
                                                                            'recons_network_config.json'))
        if self.network_mode == 'pregis':
            print("Writing {} to {}".format(self.network_config_file,
                                            os.path.join(self.network_folder, 'pregis_network_config.json')))
            os.system('cp ' + self.network_config_file + ' ' + os.path.join(self.network_folder,
                                                                            'pregis_network_config.json'))
        print("Writing {} to {}".format(self.mermaid_config_file,
                                        os.path.join(self.network_folder, 'mermaid_config.json')))
        os.system('cp ' + self.mermaid_config_file + ' ' + os.path.join(self.network_folder, 'mermaid_config.json'))

        self.log_folder = os.path.join(os.path.dirname(__file__), 'logs', '{}_net'.format(self.network_mode),
                                       my_name)
        os.system('mkdir -p ' + self.log_folder)
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

        if self.network_file:
            # resume training or loading mermaid and recons net for pregis net training

            for model_name in self.network_file:
                model_file = self.network_file[model_name]
                checkpoint = torch.load(model_file)
                if 'epoch' in checkpoint and model_name == self.network_mode + "_net":
                    current_epoch = checkpoint['epoch'] + 1
                if 'optimizer_state_dict' in checkpoint and model_name == self.network_mode + "_net":
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                try:
                    if model_name == 'pregis_net':
                        self.pregis_net.load_state_dict(checkpoint['model_state_dict'])
                        break
                    if model_name == 'mermaid_net':
                        self.pregis_net.mermaid_net.load_state_dict(checkpoint['model_state_dict'])
                    if model_name == 'recons_net':
                        self.pregis_net.recons_net.load_state_dict(checkpoint['model_state_dict'])
                except:
                    print("Model load FAILED!!!!")

        min_val_loss = 0.0

        print("Current epoch: {}".format(current_epoch))
        while current_epoch < n_epochs:
            epoch_loss_dict = {
                'mermaid_all_loss': 0.0,
                'mermaid_reg_loss': 0.0,
                'mermaid_sim_loss': 0.0,
                'recons_loss': 0.0,
                # 'segmentation_loss': 0.0,
                'all_loss': 0.0
            }

            self.scheduler.step(epoch=current_epoch + 1)
            for i, (moving_image, target_image, mask_image) in enumerate(self.train_data_loader, 0):
                self.pregis_net.train()
                self.optimizer.zero_grad()
                global_step = current_epoch * iters_per_epoch + (i + 1)

                moving_image = moving_image.cuda()
                target_image = target_image.cuda()
                mask_image = mask_image.cuda()

                self.pregis_net(moving_image, target_image, 'train')
                loss_dict = self.pregis_net.calculate_pregis_loss(moving_image, target_image, current_epoch, mask_image)
                if self.network_mode == 'pregis':
                    loss_dict['all_loss'].backward()
                else:
                    raise ValueError("Wrong network_mode")
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                self.optimizer.step()

                for loss_key in epoch_loss_dict:
                    if loss_key in loss_dict:
                        epoch_loss_dict[loss_key] += loss_dict[loss_key].item()

                if (i + 1) % summary_batch_period == 0:  # print summary every k batches
                    to_print = "====>{:0d}, {:0d}, lr:{}, all_loss:{}".format(current_epoch + 1, global_step,
                                                                              self.optimizer.param_groups[0]['lr'],
                                                                              epoch_loss_dict['all_loss'] / summary_batch_period)

                    for loss_key in epoch_loss_dict:
                        writer.add_scalar('training/training_{}'.format(loss_key),
                                          epoch_loss_dict[loss_key] / summary_batch_period, global_step=global_step)

                    images_to_show = [moving_image, target_image, mask_image]
                    phis_to_show = []

                    to_print = to_print + ", mermaid_loss:{:.6f}, sim_loss:{:.6f}, reg_loss:{:.6f}".format(
                        epoch_loss_dict['mermaid_all_loss'] / summary_batch_period,
                        epoch_loss_dict['mermaid_sim_loss'] / summary_batch_period,
                        epoch_loss_dict['mermaid_reg_loss'] / summary_batch_period
                    )
                    images_to_show.append(self.pregis_net.warped_image.detach())
                    images_to_show.append(self.pregis_net.recons.detach())
                    # images_to_show.append(self.pregis_net.abnormal_mask.detach())
                    phis_to_show.append(self.pregis_net.phi.detach())
                    to_print += ', recons_loss:{:.6f}'.format(
                        epoch_loss_dict['recons_loss'] / summary_batch_period,
                        # epoch_loss_dict['segmentation_loss'] / summary_batch_period,
                    )

                    image_summary = make_image_summary(images_to_show, phis_to_show)
                    for key, value in image_summary.items():
                        writer.add_image("training_" + key, value, global_step=global_step)

                    print(to_print)
                    epoch_loss_dict = {
                        'mermaid_all_loss': 0.0,
                        'mermaid_reg_loss': 0.0,
                        'mermaid_sim_loss': 0.0,
                        'recons_loss': 0.0,
                        # 'segmentation_loss': 0.0,
                        'all_loss': 0.0
                    }
                    writer.flush()
            if current_epoch % validate_epoch_period == 0:  # validate every k epochs
                with torch.no_grad():
                    self.pregis_net.eval()

                    eval_loss_dict = {
                        'mermaid_all_loss': 0.0,
                        'mermaid_reg_loss': 0.0,
                        'mermaid_sim_loss': 0.0,
                        'recons_loss': 0.0,
                        # 'segmentation_loss': 0.0,
                        'all_loss': 0.0
                    }
                    for j, (moving_image, target_image, mask_image) in enumerate(self.validate_data_loader, 0):
                        moving_image = moving_image.cuda()
                        target_image = target_image.cuda()
                        mask_image = mask_image.cuda()
                        self.pregis_net(moving_image, target_image, 'test')
                        loss_dict = self.pregis_net.calculate_pregis_loss(moving_image, target_image, current_epoch, mask_image)
                        for loss_key in eval_loss_dict:
                            if loss_key in loss_dict:
                                eval_loss_dict[loss_key] += loss_dict[loss_key].item()

                        if j == 0:
                            # view validation result
                            images_to_show = [moving_image, target_image, mask_image]
                            phis_to_show = []

                            images_to_show.append(self.pregis_net.warped_image.detach())
                            images_to_show.append(self.pregis_net.recons.detach())
                            # images_to_show.append(self.pregis_net.abnormal_mask.detach())

                            phis_to_show.append(self.pregis_net.phi.detach())
                            image_summary = make_image_summary(images_to_show, phis_to_show)
                            for key, value in image_summary.items():
                                writer.add_image("validation_" + key, value, global_step=global_step)
                    for loss_key in eval_loss_dict:
                        writer.add_scalar('validation/validation_{}'.format(loss_key),
                                          eval_loss_dict[loss_key] / val_iters_per_epoch, global_step=global_step)

                    if min_val_loss == 0.0 and global_step >= 50:
                        min_val_loss = eval_loss_dict['mermaid_all_loss']
                    if eval_loss_dict['mermaid_all_loss'] < min_val_loss:
                        min_val_loss = eval_loss_dict['mermaid_all_loss']
                        save_file = os.path.join(self.network_folder, 'best_eval.pth.tar')
                        print("Writing current best eval model")
                        if self.network_mode == 'pregis':
                            torch.save({'epoch': current_epoch,
                                        'model_state_dict': self.pregis_net.state_dict(),
                                        'optimizer_state_dict': self.optimizer.state_dict()},
                                       save_file)
                        else:
                            raise ValueError("Wrong Mode")

                    if current_epoch % 100 == 0:
                        save_file = os.path.join(self.network_folder, 'eval_' + str(current_epoch) + '.pth.tar')
                        if self.network_mode == 'pregis':
                            torch.save({'epoch': current_epoch,
                                        'model_state_dict': self.pregis_net.state_dict(),
                                        'optimizer_state_dict': self.optimizer.state_dict()},
                                       save_file)
                        else:
                            raise ValueError("Wrong Mode")
                    writer.flush()

                    # for kk, (moving_image, target_image) in enumerate(self.tumor_data_loader, 0):
                    #     moving_image = moving_image.cuda()
                    #     target_image = target_image.cuda()
                    #     self.pregis_net(moving_image, target_image, 'test')
                    #
                    #     if kk == 0:
                    #         # view tumor result
                    #         images_to_show = [moving_image, target_image]
                    #         phis_to_show = []
                    #
                    #         if self.network_mode == 'mermaid' or self.network_mode == 'pregis':
                    #             images_to_show.append(self.pregis_net.warped_image.detach())
                    #             phis_to_show.append(self.pregis_net.phi.detach())
                    #
                    #         if self.network_mode == 'recons' or self.network_mode == 'pregis':
                    #             images_to_show.append(self.pregis_net.recons_image.detach())
                    #             images_to_show.append(self.pregis_net.diff_image.detach())
                    #
                    #         image_summary = make_image_summary(images_to_show, phis_to_show)
                    #         for key, value in image_summary.items():
                    #             writer.add_image("tumor_" + key, value, global_step=global_step)

            current_epoch = current_epoch + 1
            writer.close()


if __name__ == '__main__':
    network = TrainPregis()
    network.train_model()
