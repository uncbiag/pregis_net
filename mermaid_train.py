import os
import datetime
import json
from utils.utils import *
import socket

from tensorboardX import SummaryWriter
from utils.visualize import make_image_summary
import torch

torch.backends.cudnn.benchmark = True


class TrainNetwork:
    def __init__(self):
        self.dataset = 'pseudo_3D'
        # network_mode selected from  'mermaid', 'recons', 'pregis'
        self.network_mode = 'mermaid'
        print("Network mode: {}".format(self.network_mode))

        self.time = None
        # specify continue training or new training
        self.is_continue = False
        self.root_folder = None
        self.__setup_root_folder()

        # set configuration file:
        self.network_config = None
        self.mermaid_config = None
        self.network_folder = None
        self.network_file = {}
        self.network_config_file = None
        self.mermaid_config_file = None
        self.log_folder = None
        self.__setup_configuration_file()

        # load models
        self.train_data_loader = None
        self.validate_data_loader = None
        self.tumor_data_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.__load_models()

        self.__setup_output_files()

    def __setup_root_folder(self):
        hostname = socket.gethostname()
        if hostname == 'biag-gpu0.cs.unc.edu':
            self.root_folder = '/playpen/xhs400/Research/pregis_net'
        elif hostname == 'biag-w05.cs.unc.edu':
            self.root_folder = '/playpen/xhs400/Research/PycharmProjects/pregis_net'
        elif 'lambda' in hostname:
            self.root_folder = '/playpen/xhs400/Research/pregis_net'
        else:
            raise ValueError("Wrong host! Please configure.")
        assert (self.root_folder is not None)

    def __setup_configuration_file(self):
        if self.is_continue:
            # to continue, specify the model folder and model
            self.network_name = "model_mermaid_time_20191022-233724_initLR_0.0005_sigma_1.732_recons_5"
            self.network_folder = os.path.join(os.path.dirname(__file__),
                                               "tmp_models/{}_net/{}".format(self.network_mode, self.network_name))
            self.log_folder = os.path.join(os.path.dirname(__file__),
                                           "logs/{}_net/{}".format(self.network_mode, self.network_name))
            self.network_config_file = os.path.join(self.network_folder, 'network_config.json')
            self.mermaid_config_file = os.path.join(self.network_folder, 'mermaid_config.json')
            self.network_file = {
                self.network_mode: os.path.join(self.network_folder, 'eval_400.pth.tar')
            }
        else:
            self.network_config_file = os.path.join(os.path.dirname(__file__),
                                                    "settings/{}/network_config.json".format(self.dataset))

            self.mermaid_config_file = os.path.join(os.path.dirname(__file__),
                                                    "settings/{}/mermaid_config.json".format(self.dataset))
            if self.network_mode == 'pregis':
                # need to load pretrained mermaid and recons
                mermaid_network_name = ""
                recons_network_name = ""
                mermaid_network_folder = os.path.join(os.path.dirname(__file__),
                                                      "tmp_models/mermaid_net/{}".format(mermaid_network_name))
                recons_network_folder = os.path.join(os.path.basename(__file__),
                                                     "tmp_models/recons_net/{}".format(recons_network_name))
                self.network_file = {
                    "mermaid": os.path.join(mermaid_network_folder, "best_eval.pth.tar"),
                    "recons": os.path.join(recons_network_folder, "best_eval.path.tar"),
                }

        with open(self.network_config_file) as f:
            self.network_config = json.load(f)

        with open(self.mermaid_config_file) as f:
            self.mermaid_config = json.load(f)
        self.network_config['model']['mermaid_config_file'] = self.mermaid_config_file

    def __load_models(self):
        self.train_data_loader, self.validate_data_loader = \
            create_dataloader(self.network_config)
        self.model = create_model(self.network_config['model'], self.network_mode)
        self.model.network_mode = self.network_mode

        self.optimizer, self.scheduler = create_optimizer(self.network_config['train'], self.model,
                                                          network_mode=self.network_mode)

    def __setup_output_files(self):
        # Setup output locations, names, etc.

        if not self.is_continue:
            now = datetime.datetime.now()
            # distinct time stamp for each model
            time = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                                  now.second)
            init_lr = self.network_config['train']['optimizer']['lr']
            my_name = "model_{}_time_{}_initLR_{}".format(self.network_mode, time, init_lr)
            sigma = self.mermaid_config['model']['registration_model']['similarity_measure']['sigma']
            my_name = my_name + '_sigma_{}'.format(sigma)

            recons_weight = self.network_config['model']['pregis_net']['recons_weight']
            my_name = my_name + '_recons_{}'.format(recons_weight)

            self.network_folder = os.path.join(os.path.dirname(__file__), 'tmp_models',
                                               '{}_net'.format(self.network_mode),
                                               my_name)
            os.system('mkdir -p ' + self.network_folder)
            if self.network_mode == 'mermaid':
                print("Writing {} to {}".format(self.network_config_file,
                                                os.path.join(self.network_folder, 'network_config.json')))
                os.system('cp ' + self.network_config_file + ' ' + os.path.join(self.network_folder,
                                                                                'network_config.json'))
            if self.network_mode == 'recons':
                print("Writing {} to {}".format(self.network_config_file,
                                                os.path.join(self.network_folder, 'network_config.json')))
                os.system('cp ' + self.network_config_file + ' ' + os.path.join(self.network_folder,
                                                                                'network_config.json'))
            if self.network_mode == 'pregis':
                print("Writing {} to {}".format(self.network_config_file,
                                                os.path.join(self.network_folder, 'network_config.json')))
                os.system('cp ' + self.network_config_file + ' ' + os.path.join(self.network_folder,
                                                                                'network_config.json'))
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

        min_val_loss = 0.0

        if self.network_file:
            # resume training or loading mermaid and recons net for pregis net training

            for model_name in self.network_file:
                # NEED MORE TEST
                model_file = self.network_file[model_name]
                checkpoint = torch.load(model_file)
                if 'min_val_loss' in checkpoint and model_name == self.network_mode:
                    min_val_loss = checkpoint['min_val_loss']
                if 'epoch' in checkpoint and model_name == self.network_mode:
                    current_epoch = checkpoint['epoch'] + 1
                if 'optimizer_state_dict' in checkpoint and model_name == self.network_mode:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                try:
                    if model_name == 'pregis':
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif model_name == 'mermaid':
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif model_name == 'recons':
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        raise ValueError("Wrong")
                except:
                    print("Model load FAILED!!!!")

        print("Current epoch: {}".format(current_epoch))
        while current_epoch < n_epochs:
            epoch_loss_dict = {
                'mermaid_all_loss': 0.0,
                'mermaid_reg_loss': 0.0,
                'mermaid_sim_loss': 0.0,
                'recons_loss': 0.0,
                'all_loss': 0.0
            }

            iters = len(self.train_data_loader)
            # self.scheduler.step(epoch=current_epoch + 1)
            for i, images in enumerate(self.train_data_loader):
                self.model.train()
                self.scheduler.step(current_epoch + i / iters)
                self.optimizer.zero_grad()
                global_step = current_epoch * iters_per_epoch + (i + 1)

                moving_image = images[0].cuda()
                target_image = images[1].cuda()
                mask_image = images[2].cuda()

                if self.network_mode == "mermaid" or self.network_mode == "pregis":
                    self.model(moving_image, target_image)
                    loss_dict = self.model.calculate_loss(moving_image, target_image, mask_image)
                elif self.network_mode == "recons":
                    self.model(moving_image, target_image)
                    loss_dict = self.model.calculate_loss(target_image, mask_image)
                else:
                    raise ValueError("Network Mode Error")
                loss_dict['all_loss'].backward()
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

                    images_to_show = [moving_image, target_image]
                    phis_to_show = []

                    if self.network_mode == "pregis" or self.network_mode == "mermaid":
                        phis_to_show.append(self.model.phi.detach())
                        images_to_show.append(self.model.warped_image.detach())
                        to_print = to_print + ", mermaid_loss:{:.6f}, sim_loss:{:.6f}, reg_loss:{:.6f}".format(
                            epoch_loss_dict['mermaid_all_loss'] / summary_batch_period,
                            epoch_loss_dict['mermaid_sim_loss'] / summary_batch_period,
                            epoch_loss_dict['mermaid_reg_loss'] / summary_batch_period
                        )

                    if self.network_mode == "pregis" or self.network_mode == "recons":
                        images_to_show.append(self.model.recons.detach())
                        to_print += ', recons_loss:{:.6f}'.format(
                            epoch_loss_dict['recons_loss'] / summary_batch_period
                        )
                    if mask_image is not None:
                        images_to_show.append(mask_image)
                    image_summary = make_image_summary(images_to_show, phis_to_show)
                    for key, value in image_summary.items():
                        writer.add_image("training_" + key, value, global_step=global_step)

                    print(to_print)
                    epoch_loss_dict = {
                        'mermaid_all_loss': 0.0,
                        'mermaid_reg_loss': 0.0,
                        'mermaid_sim_loss': 0.0,
                        'recons_loss': 0.0,
                        'all_loss': 0.0
                    }
                    writer.flush()
            if current_epoch % validate_epoch_period == 0:  # validate every k epochs
                with torch.no_grad():
                    self.model.eval()

                    eval_loss_dict = {
                        'mermaid_all_loss': 0.0,
                        'mermaid_reg_loss': 0.0,
                        'mermaid_sim_loss': 0.0,
                        'recons_loss': 0.0,
                        'all_loss': 0.0,
                        'tumor_disp_loss': 0.0,
                        'near_disp_loss': 0.0,
                        'far_disp_loss': 0.0,
                        'eval_loss': 0.0
                    }

                    for j, eval_images in enumerate(self.validate_data_loader):
                        moving_image = eval_images[0].cuda()
                        target_image = eval_images[1].cuda()
                        mask_image = eval_images[2].cuda()
                        disp_field = eval_images[3].cuda()

                        if self.network_mode == "mermaid" or self.network_mode == "pregis":
                            self.model(moving_image, target_image)
                            loss_dict = self.model.calculate_evaluation_loss(moving_image, target_image, mask_image,
                                                                             disp_field)
                        elif self.network_mode == "recons":
                            self.model(moving_image, target_image)
                            loss_dict = self.model.calculate_evaluation_loss(target_image, mask_image)
                        else:
                            raise ValueError("Network Mode Error")

                        for loss_key in eval_loss_dict:
                            if loss_key in loss_dict:
                                eval_loss_dict[loss_key] += loss_dict[loss_key].item()

                    to_print = "EVAL>{:0d}, {:0d}, eval:{:.6f}, tumor:{:.6f}, near:{:.6f}, all:{:.6f}" \
                        .format(current_epoch + 1,
                                global_step,
                                eval_loss_dict['eval_loss'] / val_iters_per_epoch,
                                eval_loss_dict['tumor_disp_loss'] / val_iters_per_epoch,
                                eval_loss_dict['near_disp_loss'] / val_iters_per_epoch,
                                eval_loss_dict['all_loss'] / val_iters_per_epoch
                                )
                    # view validation result
                    images_to_show = [moving_image, target_image]
                    phis_to_show = []

                    if self.network_mode == "pregis" or self.network_mode == "mermaid":
                        phis_to_show.append(self.model.phi.detach())
                        images_to_show.append(self.model.warped_image.detach())
                        to_print = to_print + ", sim_loss:{:.6f}".format(
                            eval_loss_dict['mermaid_sim_loss'] / val_iters_per_epoch
                        )
                    if self.network_mode == "pregis" or self.network_mode == "recons":
                        images_to_show.append(self.model.recons.detach())
                        to_print += ', recons_loss:{:.6f}'.format(
                            eval_loss_dict['recons_loss'] / val_iters_per_epoch
                        )
                    images_to_show.append(mask_image)

                    image_summary = make_image_summary(images_to_show, phis_to_show)
                    for key, value in image_summary.items():
                        writer.add_image("validation_" + key, value, global_step=global_step)
                    print(to_print)

                    for loss_key in eval_loss_dict:
                        writer.add_scalar('validation/validation_{}'.format(loss_key),
                                          eval_loss_dict[loss_key] / val_iters_per_epoch, global_step=global_step)

                    if min_val_loss == 0.0 and global_step >= 50:
                        min_val_loss = eval_loss_dict['eval_loss']
                    if eval_loss_dict['eval_loss'] < min_val_loss:
                        min_val_loss = eval_loss_dict['eval_loss']
                        save_file = os.path.join(self.network_folder, 'best_eval.pth.tar')
                        print("Writing current best eval model")
                        torch.save({'min_val_loss': min_val_loss,
                                    'epoch': current_epoch,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict()},
                                   save_file)

                    if (current_epoch + 1) % 50 == 0:
                        save_file = os.path.join(self.network_folder, 'eval_' + str(current_epoch + 1) + '.pth.tar')
                        torch.save({'min_val_loss': min_val_loss,
                                    'epoch': current_epoch,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict()},
                                   save_file)
                    writer.flush()

            current_epoch = current_epoch + 1
            writer.close()


if __name__ == '__main__':
    network = TrainNetwork()
    network.train_model()
