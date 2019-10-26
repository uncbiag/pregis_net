import torch.nn as nn
from modules.mermaid_net import MermaidNet
from modules.recons_net import ReconsNet


class PregisNet(nn.Module):
    def __init__(self, model_config):
        super(PregisNet, self).__init__()
        self.mermaid_net = MermaidNet(model_config)
        self.recons_net = ReconsNet(model_config)

        # results to return
        self.recons = None
        self.warped_image = None
        self.phi = None
        return

    def calculate_evaluation_loss(self, moving, target, normal_mask, disp_field):
        loss_from_mermaid = self.mermaid_net.calculate_evaluation_loss(moving, self.recons, normal_mask, disp_field)
        loss_from_recons = self.recons_net.calculate_evaluation_loss(target, normal_mask)
        loss_dict = {
            'eval_loss': loss_from_mermaid['eval_loss'],
            'tumor_disp_loss': loss_from_mermaid['tumor_disp_loss'],
            'near_disp_loss': loss_from_mermaid['near_disp_loss'],
            'far_disp_loss': loss_from_mermaid['far_disp_loss'],
            'recons_loss': loss_from_recons['recons_loss']
        }
        return loss_dict

    def calculate_loss(self, moving, target, normal_mask):
        loss_from_mermaid = self.mermaid_net.calculate_loss(moving, self.recons, normal_mask)
        loss_from_recons = self.recons_net.calculate_loss(target, normal_mask)
        loss_dict = {
            'recons_loss': loss_from_recons['recons_loss'],
            'mermaid_all_loss': loss_from_mermaid['mermaid_all_loss'],
            'mermaid_sim_loss': loss_from_mermaid['mermaid_sim_loss'],
            'mermaid_reg_loss': loss_from_mermaid['mermaid_reg_loss'],
            'all_loss': loss_from_mermaid['all_loss'] + loss_from_recons['all_loss']
        }
        return loss_dict

    def forward(self, moving_image, target_image):
        self.recons_net(target_image)
        self.recons = self.recons_net.recons
        self.mermaid_net(moving_image, self.recons)
        self.warped_image = self.mermaid_net.warped_image
        self.phi = self.mermaid_net.phi
        return
