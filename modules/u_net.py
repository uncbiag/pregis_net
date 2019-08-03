from modules.layers import *
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, dim=3, use_bn=False, use_dp=False, map_factor=0.5):
        super(UNet, self).__init__()
        assert(map_factor == 0.5)
        self.in11 = conv_bn_rel_dp(1,8, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.in12 = conv_bn_rel_dp(1,8, kernel_size=3,stride=1, dim=dim, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.in2 = conv_bn_rel_dp(16,16, kernel_size=3, stride=2, dim=dim, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.in3 = conv_bn_rel_dp(16,32, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.down1 = mp(kernel_size=2, dim=dim)
        self.down11 = conv_bn_rel_dp(32, 64, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.down12 = conv_bn_rel_dp(64, 64, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.down2 = mp(kernel_size=2, dim=dim)
        self.down21 = conv_bn_rel_dp(64, 128, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.down22 = conv_bn_rel_dp(128, 128, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.down3 = mp(kernel_size=2, dim=dim)
        self.bottom = conv_bn_rel_dp(128, 256, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp) 
        self.up3 = conv_bn_rel_dp(256, 128, kernel_size=2, stride=2, dim=dim, reverse=True, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        
        self.up21 = conv_bn_rel_dp(256, 128, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.up22 = conv_bn_rel_dp(128, 128, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.up2 = conv_bn_rel_dp(128, 64, kernel_size=2, stride=2, dim=dim, reverse=True, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.up11 = conv_bn_rel_dp(128, 64, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.up12 = conv_bn_rel_dp(64, 64, kernel_size=3, stride=1, dim=dim, activate_unit='leaky_unit', use_bn=use_bn, use_dp=use_dp)
        self.up1 = conv_bn_rel_dp(64, 32, kernel_size=2, stride=2, dim=dim, reverse=True, activate_unit='leaky_relu', use_bn=use_bn, use_dp=use_dp)
        self.out1 = conv_bn_rel_dp(64, 16, kernel_size=3, stride=1, dim=dim, activate_unit='None')
        self.out2 = conv_bn_rel_dp(16, dim, kernel_size=3, stride=1, dim=dim, activate_unit= 'None')


    def forward(self, m, t):
        m = self.in11(m)
        t = self.in12(t)
        x_in = torch.cat((m,t), dim=1)
        x_in = self.in2(x_in)
        x_in = self.in3(x_in)

        x_d1 = self.down1(x_in)
        x_d1 = self.down11(x_d1)
        x_d1 = self.down12(x_d1)
        
        x_d2 = self.down2(x_d1)
        x_d2 = self.down21(x_d2)
        x_d2 = self.down22(x_d2)
      
        x_bt = self.down3(x_d2)
        x_bt = self.bottom(x_bt)
        x_u2 = self.up3(x_bt)

        x_u2 = torch.cat((x_d2, x_u2), dim=1)
        x_u2 = self.up21(x_u2)
        x_u2 = self.up22(x_u2)
        x_u1 = self.up2(x_u2)

        x_u1 = torch.cat((x_d1, x_u1), dim=1)
        x_u1 = self.up11(x_u1)
        x_u1 = self.up12(x_u1)
        x_out = self.up1(x_u1)
      
        x_out = torch.cat((x_in, x_out), dim=1)
        x_out = self.out1(x_out)
        x_out = self.out2(x_out)
  
        return x_out 
