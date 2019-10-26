import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import sys
import os
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../mermaid'))
import pyreg.fileio as py_fio

from modules.pregis_net import PregisNet
from modules.mermaid_net import MermaidNet
from modules.recons_net import ReconsNet
from torch.utils.data import DataLoader

from data_loaders import pseudo_2D as pseudo_2D_dataset
from data_loaders import pseudo_3D as pseudo_3D_dataset
from data_loaders import brats_3D as brats_3D_dataset


class AdamW(optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{i}}\pi))

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0`(after restart), set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
        self.T_cur = self.last_epoch

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class JointScheduler(object):
    def __init__(self, mermaid_scheduler, recons_scheduler):
        self.mermaid_scheduler = mermaid_scheduler
        self.recons_scheduler = recons_scheduler

    def step(self, epoch=None):
        self.mermaid_scheduler.step(epoch)
        self.recons_scheduler.step(epoch)


class JointOptimizer(object):
    def __init__(self, mermaid_optimizer, recons_optimizer):
        self.mermaid_optimizer = mermaid_optimizer
        self.recons_optimizer = recons_optimizer

    def zero_grad(self):
        self.mermaid_optimizer.zero_grad()
        self.recons_optimizer.zero_grad()

    def step(self):
        self.mermaid_optimizer.step()
        self.recons_optimizer.step()

    def load_state_dict(self, mermaid_opt_dict, recons_opt_dict):
        self.mermaid_optimizer.load_state_dict(mermaid_opt_dict)
        self.recons_optimizer.load_state_dict(recons_opt_dict)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def create_model(network_config, network_mode):
    if network_mode == 'pregis':
        model = PregisNet(network_config)
    elif network_mode == 'mermaid':
        model = MermaidNet(network_config)
    elif network_mode == 'recons':
        model = ReconsNet(network_config)
    model.cuda()
    model.apply(weights_init)
    return model


def create_dataset(dataset_name, dataset_type, mode):
    if dataset_name == "brats_3D":
        dataset = brats_3D_dataset.Brats3DDataset
        return dataset(mode)
    elif dataset_name == "pseudo_2D":
        dataset = pseudo_2D_dataset.Pseudo2DDataset
        return dataset(dataset_type, mode)
    elif dataset_name == "pseudo_3D":
        dataset = pseudo_3D_dataset.Pseudo3DDataset
        return dataset(dataset_type, mode)
    else:
        raise ValueError("dataset not available")


def create_dataloader(network_config):
    model_config = network_config['model']

    train_dataset = network_config['train']['dataset']
    train_dataset_type = network_config['train']['dataset_type']

    validate_dataset = network_config['validate']['dataset']
    validate_dataset_type = network_config['validate']['dataset_type']

    my_train_dataset = create_dataset(train_dataset, train_dataset_type, 'train')
    my_validate_dataset = create_dataset(validate_dataset, validate_dataset_type, 'validate')
    atlas_file = my_train_dataset.atlas_file
    print(atlas_file)
    image_io = py_fio.ImageIO()
    target_image, target_hdrc, target_spacing, _ = image_io.read_to_nc_format(atlas_file, silent_mode=True)

    train_data_loader = DataLoader(my_train_dataset,
                                   batch_size=network_config['train']['batch_size'],
                                   shuffle=True, num_workers=1,
                                   drop_last=True)
    validate_data_loader = DataLoader(my_validate_dataset,
                                      batch_size=network_config['train']['batch_size'],
                                      shuffle=True,
                                      num_workers=1,
                                      drop_last=False)
    model_config['img_sz'] = [network_config['train']['batch_size'], 1] + list(target_image.shape[2:])
    model_config['dim'] = len(target_image.shape[2:])
    model_config['target_hdrc'] = target_hdrc
    model_config['target_spacing'] = target_spacing
    print(model_config['target_spacing'])
    return train_data_loader, validate_data_loader


def create_optimizer(config, model, network_mode=None):
    method = config['optimizer']['name']
    scheduler_type = config['optimizer']['scheduler_type']
    lr = config['optimizer']['lr']
    if method == "ADAM":
        optimizer_to_use = optim.Adam
    elif method == "ADAMW":
        optimizer_to_use = AdamW
    else:
        raise ValueError("optimizer not supported")

    if network_mode == 'pregis':
        mermaid_optimizer = optimizer_to_use(model.mermaid_net.parameters(), lr=lr)
        recons_optimizer = optimizer_to_use(model.recons_net.parameters(), lr=lr)
        optimizer = JointOptimizer(mermaid_optimizer, recons_optimizer)
    else:
        optimizer = optimizer_to_use(model.parameters(), lr=lr)

    if scheduler_type == 'multistepLR':
        milestones = list(config['optimizer']['scheduler']['milestones'])
        gamma = config['optimizer']['scheduler']['gamma']
        if network_mode == 'pregis':
            mermaid_scheduler = lr_scheduler.MultiStepLR(optimizer.mermaid_optimizer, milestones=milestones, gamma=gamma)
            recons_scheduler = lr_scheduler.MultiStepLR(optimizer.recons_optimizer, milestones=milestones, gamma=gamma)
            scheduler = JointScheduler(mermaid_scheduler, recons_scheduler)
        else:
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        T_0 = config['optimizer']['scheduler']['CosineAnnealingWarmRestarts']['T_0']
        T_mult = config['optimizer']['scheduler']['CosineAnnealingWarmRestarts']['T_mult']
        if network_mode == 'pregis':
            mermaid_scheduler = CosineAnnealingWarmRestarts(optimizer.mermaid_optimizer, T_0=T_0, T_mult=T_mult)
            recons_scheduler = CosineAnnealingWarmRestarts(optimizer.recons_optimizer, T_0=T_0, T_mult=T_mult)
            scheduler = JointScheduler(mermaid_scheduler, recons_scheduler)
        else:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)

    elif scheduler_type == 'CosineAnnealing':
        T_max = config['n_of_epochs']
        if network_mode == 'pregis':
            mermaid_scheduler = lr_scheduler.CosineAnnealingLR(optimizer.mermaid_optimizer, T_max=T_max)
            recons_scheduler = lr_scheduler.CosineAnnealingLR(optimizer.recons_optimizer, T_max=T_max)
            scheduler = JointScheduler(mermaid_scheduler, recons_scheduler)
        else:
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError("scheduler not supported")
    return optimizer, scheduler


def create_loss(config):
    name = config['loss']['name']
    reduction = config['loss']['reduction']

    if name == 'MSE':
        criterion = nn.MSELoss(reduction=reduction).cuda()
    elif name == 'L1':
        criterion = nn.L1Loss(reduction=reduction).cuda()
    else:
        print('Loss specified unrecognized. MSE Loss is used')
        criterion = nn.MSELoss(reduction=reduction).cuda()
    return criterion


if __name__ == '__main__':
    # This is to test the results from evaluate_model
    print('test')
