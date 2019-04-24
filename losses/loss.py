import torch
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self, _input, target):
        x = _input - target
        dim = len(x.shape)-2
        batch_size = x.size()[0]
        img_sz = x.size()[2:]
        grad = torch.cuda.FloatTensor(batch_size, dim, *img_sz).fill_(0)
        if dim == 1:
            grad[:,0,1:-1] = (x[:,0,2:] - x[:,0,:-2])/2.
        elif dim == 2:
            grad[:,0,1:-1,:] = (x[:,0,2:,:] - x[:,0,:-2,:])/2.
            grad[:,1,:,1:-1] = (x[:,0,:,2:] - x[:,0,:,:-2])/2.
        elif dim == 3:
            grad[:,0,1:-1,:,:] = (x[:,0,2:,:,:] - x[:,0,:-2,:,:])/2.
            grad[:,1,:,1:-1,:] = (x[:,0,:,2:,:] - x[:,0,:,:-2,:])/2.
            grad[:,2,:,:,1:-1] = (x[:,0,:,:,2:] - x[:,0,:,:,:-2])/2.
        else:
            raise ValueError("dim error")
        grad_n = torch.norm(grad, p=2, dim=1, keepdim=True)
        return torch.mean(grad_n)
        #return torch.sum(grad_n)



if __name__ == '__main__':
    x = torch.rand(1,1,10,10,10)
    y = torch.rand(1,1,10,10,10)
    loss = TVLoss()
    grad = loss(x,y)
    print(torch.mean(torch.abs(x-y)))
    print(grad)
