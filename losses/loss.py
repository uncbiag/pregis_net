import torch
import torch.nn as nn


class GMLoss(nn.Module):
    # Geman & McClure Loss:
    # L(x,\sigma) = \frac{x^2}{\simga + x^2}, \sigma>0
    # L(0,\sigma) = 0, L(\inf, \sigma) = 1
    def __init__(self, sigma=0.01):
        super(GMLoss, self).__init__()


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
            grad[:, 0, :-1] = x[:, 0, 1:]-x[:, 0, :-1]
        elif dim == 2:
            grad[:, 0, :-1, :] = x[:, 0, 1:, :] - x[:, 0, :-1, :]
            grad[:, 1, :, :-1] = x[:, 0, :, 1:] - x[:, 0, :, :-1]
        elif dim == 3:
            grad[:, 0, :-1, :, :] = x[:, 0, 1:, :, :] - x[:, 0, :-1, :, :]
            grad[:, 1, :, :-1, :] = x[:, 0, :, 1:, :] - x[:, 0, :, :-1, :]
            grad[:, 2, :, :, :-1] = x[:, 0, :, :, 1:] - x[:, 0, :, :, :-1]
        else:
            raise ValueError("dim error")
        grad_n = torch.norm(grad, p=2, dim=1, keepdim=True)
        return torch.mean(grad_n)


if __name__ == '__main__':
    x = torch.rand(1, 1, 10, 10, 10)
    y = torch.rand(1, 1, 10, 10, 10)
    loss = TVLoss()
    grad = loss(x, y)
    print(torch.mean(torch.abs(x-y)))
    print(grad)
