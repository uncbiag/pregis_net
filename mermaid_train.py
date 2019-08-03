import os
import datetime
import json
from utils.utils import *



def train_network():
    dataset = 'pseudo_3D'
    now = datetime.datetime.now()
    my_time = "{:04d}{:02d}{:02d}-{:02d}{:02d}{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                             now.second)
    is_continue = False
    model_folder = None

    if is_continue:
        model_folder = os.path.join(os.path.dirname(__file__), "/main/tmp_models/model_20190324-011739")
        my_name = model_folder.split('/')[-1]
    if model_folder is not None:
        network_config_file = os.path.join(model_folder, 'network_config.json')
        mermaid_config_file = os.path.join(model_folder, 'mermaid_config.json')
    else:
        network_config_file = os.path.join(os.path.dirname(__file__), "settings/{}/network_config.json".format(dataset))
        mermaid_config_file = os.path.join(os.path.dirname(__file__), 'settings/{}/mermaid_config.json'.format(dataset))
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
    model = create_model(model_config)
    optimizer, scheduler = create_optimizer(train_config, model)

    # criterion = create_loss(train_config)
    if not is_continue:
        my_name = "model_{}_sigma{:.3f}_lr{}".format(
            my_time,
            sigma,
            init_lr)
        model_folder = os.path.join(os.path.dirname(__file__), 'tmp_models', 'vae', my_name)
        os.system('mkdir -p ' + model_folder)
        os.system('cp ' + network_config_file + ' ' + model_folder)
        os.system('cp ' + mermaid_config_file + ' ' + model_folder)
        log_folder = os.path.join(os.path.dirname(__file__), 'logs', my_name)
        os.system('mkdir -p ' + log_folder)

    train_model(model, train_data_loader, validate_data_loader, optimizer, scheduler, network_config, my_name,
                model_folder, model_path=model_path)

    return


if __name__ == '__main__':
    train_network()