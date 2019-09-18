import os
import datetime
import json
from utils.utils import *
import socket

from tensorboardX import SummaryWriter
from utils.visualize import make_image_summary


class TrainPregis:

    def __init__(self):
        self.dataset = 'pseudo_2D'
        # network_mode selected from  'mermaid', 'recons', 'pregis'
        self.network_mode = 'recons'

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
            if self.network_mode == 'mermaid':
                self.network_folder = ""
                self.network_file['mermaid_net'] = os.path.join(self.network_folder, "")
                self.network_config_file = os.path.join(self.network_folder, 'mermaid_network_config.json')
            if self.network_mode == 'recons':
                self.network_folder = os.path.join(self.root_folder,
                                                   'tmp_models/recons_net',
                                                   'model_recons_time_20190913-163436_initLR_0.0005_kld_0.01_recons_1_sim_0.1_useTV_False_tv_0.1')
                self.network_file['recons_net'] = os.path.join(self.network_folder, "eval_99.pth.tar")
                self.network_config_file = os.path.join(self.network_folder, 'recons_network_config.json')
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
            if self.network_mode == 'pregis':
                # needs to specify which mermaid net and recons net pretrained model to load
                # otherwise load from scratch
                mermaid_net_folder = os.path.join(self.root_folder, 'tmp_models/mermaid_net',
                                                  "model_mermaid_time_20190911-160624_initLR_0.0005_sigma_1.732")
                if mermaid_net_folder != "":
                    self.network_file['mermaid_net'] = os.path.join(mermaid_net_folder, "eval_1.pth.tar")
                    assert (os.path.isfile(self.network_file['mermaid_net']))

                recons_net_folder = os.path.join(self.root_folder, 'tmp_models/recons_net',
                                                 "model_recons_time_20190911-115036_initLR_0.0005_kld_0.001_recons_1_useTV_False_tv_0.1")
                if recons_net_folder != "":
                    self.network_file['recons_net'] = os.path.join(recons_net_folder, "eval_2.pth.tar")
                    assert (os.path.isfile(self.network_file['recons_net']))

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
        elif self.network_mode == 'mermaid':
            self.optimizer, self.scheduler = create_optimizer(train_config, self.pregis_net.mermaid_net)
        elif self.network_mode == 'recons':
            self.optimizer, self.scheduler = create_optimizer(train_config, self.pregis_net.recons_net)
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
        if self.network_mode == 'mermaid' or self.network_mode == 'pregis':
            my_name = my_name + '_sigma_{}'.format(sigma)

        if self.network_mode == 'recons' or self.network_mode == 'pregis':
            print(self.network_config)
            use_tv_loss = self.network_config['model']['pregis_net']['recons_net']['use_TV_loss']

            kld_weight = self.network_config['model']['pregis_net']['recons_net']['KLD_weight']
            tv_weight = self.network_config['model']['pregis_net']['recons_net']['TV_weight']
            recons_weight = self.network_config['model']['pregis_net']['recons_net']['recons_weight']
            sim_weight = self.network_config['model']['pregis_net']['recons_net']['sim_weight']
            my_name = my_name + '_kld_{}_recons_{}_sim_{}_useTV_{}_tv_{}'.format(kld_weight,
                                                                                 recons_weight,
                                                                                 sim_weight,
                                                                                 use_tv_loss,
                                                                                 tv_weight)

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
                'vae_kld_loss': 0.0,
                'recons_loss_l1': 0.0,
                'recons_loss_TV': 0.0,
                'vae_recons_loss': 0.0,
                'vae_sim_loss': 0.0,
                'vae_all_loss': 0.0,
                'all_loss': 0.0
            }

            self.scheduler.step(epoch=current_epoch + 1)
            for i, (moving_image, target_image) in enumerate(self.train_data_loader, 0):
                self.pregis_net.train()
                self.optimizer.zero_grad()
                global_step = current_epoch * iters_per_epoch + (i + 1)

                moving_image = moving_image.cuda()
                target_image = target_image.cuda()

                self.pregis_net(moving_image, target_image)
                loss_dict = self.pregis_net.cal_pregis_loss(moving_image, target_image, current_epoch)
                if self.network_mode == 'pregis':
                    loss_dict['all_loss'].backward()
                elif self.network_mode == 'mermaid':
                    loss_dict['mermaid_all_loss'].backward()
                elif self.network_mode == 'recons':
                    loss_dict['vae_all_loss'].backward()
                else:
                    raise ValueError("Wrong network_mode")

                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                self.optimizer.step()

                for loss_key in epoch_loss_dict:
                    if loss_key in loss_dict:
                        epoch_loss_dict[loss_key] += loss_dict[loss_key].item()

                if (i + 1) % summary_batch_period == 0:  # print summary every k batches
                    to_print = "====>{:0d}, {:0d}, lr:{}".format(current_epoch + 1, global_step,
                                                                 self.optimizer.param_groups[0]['lr'])

                    for loss_key in epoch_loss_dict:
                        writer.add_scalar('training/training_{}'.format(loss_key),
                                          epoch_loss_dict[loss_key] / summary_batch_period, global_step=global_step)

                    images_to_show = [moving_image, target_image]
                    phis_to_show = []
                    if self.network_mode == 'mermaid' or self.network_mode == 'pregis':
                        images_to_show.append(self.pregis_net.warped_image.detach())
                        phis_to_show.append(self.pregis_net.phi.detach())

                        to_print = to_print + ", mermaid_loss:{:.6f}, sim_loss:{:.6f}, reg_loss:{:.6f}".format(
                            epoch_loss_dict['mermaid_all_loss'] / summary_batch_period,
                            epoch_loss_dict['mermaid_sim_loss'] / summary_batch_period,
                            epoch_loss_dict['mermaid_reg_loss'] / summary_batch_period
                        )

                    if self.network_mode == 'recons' or self.network_mode == 'pregis':
                        images_to_show.append(self.pregis_net.recons_image.detach())
                        images_to_show.append(self.pregis_net.diff_image.detach())
                        to_print += ', vae_loss:{:.6f}, vae_sim_loss:{:.6f}, recons_loss:{:.6f}, l1_loss:{:.6f}, tv_loss:{:.6f}, kld_loss:{:.6f}'.format(
                            epoch_loss_dict['vae_all_loss'] / summary_batch_period,
                            epoch_loss_dict['vae_sim_loss'] / summary_batch_period,
                            epoch_loss_dict['vae_recons_loss'] / summary_batch_period,
                            epoch_loss_dict['recons_loss_l1'] / summary_batch_period,
                            epoch_loss_dict['recons_loss_TV'] / summary_batch_period,
                            epoch_loss_dict['vae_kld_loss'] / summary_batch_period
                        )

                    image_summary = make_image_summary(images_to_show, phis_to_show)
                    for key, value in image_summary.items():
                        writer.add_image("training_" + key, value, global_step=global_step)

                    print(to_print)
                    epoch_loss_dict = {
                        'mermaid_all_loss': 0.0,
                        'mermaid_reg_loss': 0.0,
                        'mermaid_sim_loss': 0.0,
                        'vae_kld_loss': 0.0,
                        'recons_loss_l1': 0.0,
                        'recons_loss_TV': 0.0,
                        'vae_recons_loss': 0.0,
                        'vae_sim_loss': 0.0,
                        'vae_all_loss': 0.0,
                        'all_loss': 0.0
                    }

            if current_epoch % validate_epoch_period == 0:  # validate every k epochs
                with torch.no_grad():
                    self.pregis_net.eval()

                    eval_loss_dict = {
                        'mermaid_all_loss': 0.0,
                        'mermaid_reg_loss': 0.0,
                        'mermaid_sim_loss': 0.0,
                        'vae_kld_loss': 0.0,
                        'recons_loss_l1': 0.0,
                        'recons_loss_TV': 0.0,
                        'vae_recons_loss': 0.0,
                        'vae_sim_loss': 0.0,
                        'vae_all_loss': 0.0,
                        'all_loss': 0.0
                    }
                    for j, (moving_image, target_image) in enumerate(self.validate_data_loader, 0):
                        moving_image = moving_image.cuda()
                        target_image = target_image.cuda()
                        self.pregis_net(moving_image, target_image)
                        loss_dict = self.pregis_net.cal_pregis_loss(moving_image, target_image, current_epoch)
                        for loss_key in eval_loss_dict:
                            if loss_key in loss_dict:
                                eval_loss_dict[loss_key] += loss_dict[loss_key].item()

                        if j == 0:
                            # view validation result
                            images_to_show = [moving_image, target_image]
                            phis_to_show = []

                            if self.network_mode == 'mermaid' or self.network_mode == 'pregis':
                                images_to_show.append(self.pregis_net.warped_image.detach())
                                phis_to_show.append(self.pregis_net.phi.detach())

                            if self.network_mode == 'recons' or self.network_mode == 'pregis':
                                images_to_show.append(self.pregis_net.recons_image.detach())
                                images_to_show.append(self.pregis_net.diff_image.detach())

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
                        elif self.network_mode == 'mermaid':
                            torch.save({'epoch': current_epoch,
                                        'model_state_dict': self.pregis_net.mermaid_net.state_dict(),
                                        'optimizer_state_dict': self.optimizer.state_dict()},
                                       save_file)
                        elif self.network_mode == 'recons':
                            torch.save({'epoch': current_epoch,
                                        'model_state_dict': self.pregis_net.recons_net.state_dict(),
                                        'optimizer_state_dict': self.optimizer.state_dict()},
                                       save_file)
                        else:
                            raise ValueError("Wrong Mode")

                    save_file = os.path.join(self.network_folder, 'eval_' + str(current_epoch) + '.pth.tar')
                    if self.network_mode == 'pregis':
                        torch.save({'epoch': current_epoch,
                                    'model_state_dict': self.pregis_net.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict()},
                                   save_file)
                    elif self.network_mode == 'mermaid':
                        torch.save({'epoch': current_epoch,
                                    'model_state_dict': self.pregis_net.mermaid_net.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict()},
                                   save_file)
                    elif self.network_mode == 'recons':
                        torch.save({'epoch': current_epoch,
                                    'model_state_dict': self.pregis_net.recons_net.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict()},
                                   save_file)
                    else:
                        raise ValueError("Wrong Mode")



                    for kk, (moving_image, target_image) in enumerate(self.tumor_data_loader, 0):
                        moving_image = moving_image.cuda()
                        target_image = target_image.cuda()
                        self.pregis_net(moving_image, target_image)

                        if kk == 0:
                            # view tumor result
                            images_to_show = [moving_image, target_image]
                            phis_to_show = []

                            if self.network_mode == 'mermaid' or self.network_mode == 'pregis':
                                images_to_show.append(self.pregis_net.warped_image.detach())
                                phis_to_show.append(self.pregis_net.phi.detach())

                            if self.network_mode == 'recons' or self.network_mode == 'pregis':
                                images_to_show.append(self.pregis_net.recons_image.detach())
                                images_to_show.append(self.pregis_net.diff_image.detach())

                            image_summary = make_image_summary(images_to_show, phis_to_show)
                            for key, value in image_summary.items():
                                writer.add_image("tumor_" + key, value, global_step=global_step)

            current_epoch = current_epoch + 1
            writer.close()


if __name__ == '__main__':
    network = TrainPregis()
    network.train_model()
