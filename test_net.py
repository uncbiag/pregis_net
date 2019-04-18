from modules.unet import ReconsNet
from modules.pregis_net import PregisNet
from modules.vae import VaeNet

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import pyreg.fileio as py_fio
def test_unet_blocks():
    blk = down_conv(in_ch=3, out_ch=64, dim=3, use_bn=False, use_dp=False)
    return blk

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data,0)

def test_vae():
    model = VaeNet(dim=2)
    model.cuda()

    t = torch.FloatTensor(10,1,192,192).uniform_().cuda()
    o, mu, logvar = model(t)
    print(t.shape)
    print(o.shape)
    print(mu.shape)
    print(logvar.shape)
    return

def test_unet():
    model = ReconsNet(n_ft=16)
    model.cuda()
    model.apply(weights_init)
    criterion = nn.MSELoss(reduction='mean').cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for i in range(1000):    
        model.train()
        optimizer.zero_grad()
  
        #moving_img = np.random.randn(1,1,256,256,64)
        #target_img = np.random.randn(1,1,256,256,64)

        #moving_var = torch.from_numpy(moving_img).float().cuda()
        #target_var = torch.from_numpy(target_img).float().cuda()

        m = torch.FloatTensor(1,1,192,192).uniform_().cuda()
        t = torch.FloatTensor(1,1,192,192).uniform_().cuda()
        o_m, o_i = model(m, t)

        m0 = torch.FloatTensor(1,3,96,96).fill_(1).cuda()
        i0 = torch.FloatTensor(1,1,192,192).uniform_().cuda()
        loss = criterion(o_m, m0) + criterion(o_i, i0)
        loss.backward()
        optimizer.step()
        print(loss.item())
    return


def test_pregis_net():
    m = torch.FloatTensor(1,1,256,256,128).uniform_().cuda()
    t = torch.FloatTensor(1,1,256,256,128).uniform_().cuda()

    m0 = torch.FloatTensor(1,3,256,256,128).fill_(1).cuda()
    i0 = torch.FloatTensor(1,1,256,256,128).uniform_().cuda()

    model = PregisNet(t.size())
    model.cuda()

    recons, phi, warped = model(m, t)

    print(recons.size())
    print(phi.size())
    print(warped.size())

if __name__ == '__main__':
    test_vae()

