import os
import datetime
import json
from utils.utils import *
import glob

from tensorboardX import SummaryWriter
from utils.visualize import make_image_summary


def train_model(model, train_data_loader, validate_data_loader, optimizer, scheduler,
                network_config, my_name, my_model_folder, model_path=None):
    n_epochs = network_config['train']['n_of_epochs']
    current_epoch = 0
    batch_size = network_config['train']['batch_size']
    iters_per_epoch = len(train_data_loader.dataset) // batch_size
    val_iters_per_epoch = len(validate_data_loader.dataset) // batch_size
    summary_batch_period = min(network_config['train']['min_summary_period'], iters_per_epoch)
    validate_epoch_period = network_config['validate']['validate_epoch_period']

    print('batch_size:', str(batch_size))
    print('iters_per_epoch:', str(iters_per_epoch))
    print('val_iters_per_epoch:', str(val_iters_per_epoch))
    print('summary_batch_period:', str(summary_batch_period))
    print("validate_epoch_period:", str(validate_epoch_period))

    writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 'logs', 'mermaid_net', my_name))

    if model_path is not None:
        # resume training
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            if 'epoch' in checkpoint:
                current_epoch = checkpoint['epoch'] + 1
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                print("Model load FAILED")
            if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    min_val_loss = 0.0

    while current_epoch < n_epochs:
        epoch_loss_dict = {
            'mermaid_all_loss': 0.0,
            'mermaid_sim_loss': 0.0,
            'mermaid_reg_loss': 0.0,
        }

        scheduler.step(epoch=current_epoch + 1)
        for i, (moving_image, target_image) in enumerate(train_data_loader, 0):
            model.train()
            optimizer.zero_grad()
            global_step = current_epoch * iters_per_epoch + (i + 1)

            moving_image = moving_image.cuda()
            target_image = target_image.cuda()

            model(moving_image, target_image)
            loss_dict = model.cal_mermaid_loss(moving_image, target_image)
            all_loss = loss_dict['mermaid_all_loss']
            all_loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            for loss_key in epoch_loss_dict:
                if loss_key in loss_dict:
                    epoch_loss_dict[loss_key] += loss_dict[loss_key].item()
                else:
                    epoch_loss_dict[loss_key] += 0.0

            if (i + 1) % summary_batch_period == 0:  # print summary every k batches
                print('====>{:0d}, {:0d}, loss:{:.6f}, sim_loss:{:.6f}, reg_loss:{:.6f}'.format(
                    current_epoch + 1,
                    global_step,
                    epoch_loss_dict['mermaid_all_loss'] / summary_batch_period,
                    epoch_loss_dict['mermaid_sim_loss'] / summary_batch_period,
                    epoch_loss_dict['mermaid_reg_loss'] / summary_batch_period)
                )

                for loss_key in epoch_loss_dict:
                    writer.add_scalar('training/training_{}'.format(loss_key),
                                      epoch_loss_dict[loss_key] / summary_batch_period, global_step=global_step)

                images_to_show = []
                images_to_show.append(moving_image)
                images_to_show.append(target_image)
                images_to_show.append(model.warped.detach())

                phis_to_show = []
                phis_to_show.append(model.phi.detach())

                image_summary = make_image_summary(images_to_show, phis_to_show)
                for key, value in image_summary.items():
                    writer.add_image("training_" + key, value, global_step=global_step)
                epoch_loss_dict = {
                    'mermaid_all_loss': 0.0,
                    'mermaid_sim_loss': 0.0,
                    'mermaid_reg_loss': 0.0,
                }

        if current_epoch % validate_epoch_period == 0:  # validate every k epochs
            with torch.no_grad():
                model.eval()
                eval_loss_dict = {
                    'mermaid_all_loss': 0.0,
                    'mermaid_reg_loss': 0.0,
                    'mermaid_sim_loss': 0.0,
                }
                for j, (moving_image, target_image) in enumerate(validate_data_loader, 0):
                    moving_image = moving_image.cuda()
                    target_image = target_image.cuda()
                    model(moving_image, target_image)
                    loss_dict = model.cal_mermaid_loss(moving_image, target_image)
                    for loss_key in eval_loss_dict:
                        if loss_key in loss_dict:
                            eval_loss_dict[loss_key] += loss_dict[loss_key].item()
                        else:
                            eval_loss_dict[loss_key] += 0.0

                print('=EVAL>{:0d}, {:0d}, loss:{:.6f}, sim_loss:{:.6f}, reg_loss:{:.6f}'.format(
                    current_epoch + 1,
                    global_step,
                    eval_loss_dict['mermaid_all_loss'] / val_iters_per_epoch,
                    eval_loss_dict['mermaid_sim_loss'] / val_iters_per_epoch,
                    eval_loss_dict['mermaid_reg_loss'] / val_iters_per_epoch)
                )
                for loss_key in eval_loss_dict:
                    writer.add_scalar('validation/validation_{}'.format(loss_key),
                                      eval_loss_dict[loss_key] / val_iters_per_epoch, global_step=global_step)

                if min_val_loss == 0.0 and global_step >= 50:
                    min_val_loss = eval_loss_dict['mermaid_all_loss']
                if eval_loss_dict['mermaid_all_loss'] <= min_val_loss:
                    min_val_loss = eval_loss_dict['mermaid_all_loss']
                    save_file = os.path.join(my_model_folder, 'best_eval.pth.tar')
                    print("Writing current best eval model")
                    torch.save({'epoch': current_epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}, save_file)
                save_file = os.path.join(my_model_folder, 'eval_' + str(current_epoch) + '.pth.tar')
                torch.save({'epoch': current_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, save_file)


                images_to_show = []
                images_to_show.append(moving_image)
                images_to_show.append(target_image)
                images_to_show.append(model.warped.detach())

                phis_to_show = []
                phis_to_show.append(model.phi.detach())

                image_summary = make_image_summary(images_to_show, phis_to_show)
                for key, value in image_summary.items():
                    writer.add_image("validation_" + key, value, global_step=global_step)
        current_epoch = current_epoch + 1

def train_network():
    dataset = 'pseudo_3D'
    sim = 'ncc'
    now = datetime.datetime.now()
    my_time = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                             now.second)
    is_continue = False
    model_folder = None

    if is_continue:
        #model_folder = os.path.join(os.path.dirname(__file__), "/main/tmp_models/model_20190324-011739")
        my_name = model_folder.split('/')[-1]
    if model_folder is not None:
        network_config_file = os.path.join(model_folder, 'network_config.json')
        for file in glob.glob(os.path.join(model_folder, 'mermaid_config_*.json')):
            mermaid_config_file = file
    else:
        network_config_file = os.path.join(os.path.dirname(__file__), "settings/{}/network_config.json".format(dataset))
        mermaid_config_file = os.path.join(os.path.dirname(__file__), 'settings/{}/mermaid_config_' + sim + ' .json'.format(dataset))
    with open(mermaid_config_file) as f:
        mermaid_config = json.load(f)
    with open(network_config_file) as f:
        network_config = json.load(f)


    sigma = mermaid_config['model']['registration_model']['similarity_measure']['sigma']
    init_lr = network_config['train']['optimizer']['lr']
    model_path = None

    model_config = network_config['model']
    train_config = network_config['train']
    validate_config = network_config['validate']

    train_data_loader, validate_data_loader, _ = create_dataloader(model_config, train_config, validate_config)
    model_config['mermaid_config_file'] = mermaid_config_file
    model = create_model(model_config, model_name='mermaid_net')
    optimizer, scheduler = create_optimizer(train_config, model)

    # criterion = create_loss(train_config)
    if not is_continue:
        my_name = "model_{}_sigma{:.3f}_lr{}".format(
            my_time,
            sigma,
            init_lr)
        model_folder = os.path.join(os.path.dirname(__file__), 'tmp_models', 'mermaid_net', my_name)
        os.system('mkdir -p ' + model_folder)
        os.system('cp ' + network_config_file + ' ' + model_folder)
        os.system('cp ' + mermaid_config_file + ' ' + model_folder)
        log_folder = os.path.join(os.path.dirname(__file__), 'logs', 'mermaid_net', my_name)
        os.system('mkdir -p ' + log_folder)

    train_model(model, train_data_loader, validate_data_loader, optimizer, scheduler, network_config, my_name, model_folder, model_path=model_path)

    return


if __name__ == '__main__':
    train_network()