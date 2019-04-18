import torch
import torch.nn as nn

class LNCCLoss(nn.Module):
    def __init__(self, spacing, params):
        super(LNCCLoss,self).__init__(spacing,params)
        self.dim = len(spacing)
        self.resol_bound = params['similarity_measure']['lncc'][('resol_bound',[128,64], "resolution bound for using different strategy")]
        self.kernel_size_ratio = params['similarity_measure']['lncc'][('kernel_size_ratio',[[1./16,1./8,1./4],[1./4,1./2],[1./2]], "kernel size, ratio of input size")]
        self.kernel_weight_ratio = params['similarity_measure']['lncc'][('kernel_weight_ratio',[[0.1, 0.3, 0.6],[0.3,0.7],[1.]], "kernel size, ratio of input size")]
        self.stride = params['similarity_measure']['lncc'][('stride',[1./4,1./4,1./4], "step size, responded with ratio of kernel size")]
        self.dilation = params['similarity_measure']['lncc'][('dilation',[1,2,2], "dilation param, responded with ratio of kernel size")]
        if self.resol_bound[0] >-1:
            assert len(self.resol_bound)+1 == len(self.kernel_size_ratio)
            assert len(self.resol_bound)+1 == len(self.kernel_weight_ratio)
            assert max(len(kernel) for kernel in self.kernel_size_ratio) == len(self.stride)
            assert max(len(kernel) for kernel in self.kernel_size_ratio) == len(self.dilation)

    def compute_similarity_multiC(self, I0, I1, I0Source=None, phi=None):
        sz0 = I0.size()[0]
        sz1 = I1.size()[0]
        assert(sz0 == sz1 or sz0 == sz1 - 1)

        # last channel of target image is similarity mask
        num_of_labels = sz0-1
        mask = None
        if (sz0 == sz1 - 1):
            mask = I1[-1,...]

        sim = 0.0
        sim = sim + self.compute_similarity(I0[0,...], I1[0,...], isLabel=False, similarity_mask=None)

        if num_of_labels > 0:
            I0_labels = I0[1:,...]
            I1_labels = I1[1:,...]
            for nrL in range(num_of_labels):
                sim = sim + self.compute_similarity(I0_labels[nrL,...], I1_labels[nrL,...], isLabel=True, similarity_mask=mask)

        return sim/self.sigma**2




class TVLoss(nn.Module):
    # 2D TV Loss
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self, _input, target):
        x = _input - target
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        h_tv = torch.zeros_like(x)
        w_tv = torch.zeros_like(x)
        h_tv[:,:,1:,:] = (x[:,:,1:,:]-x[:,:,:-1,:])**2
        w_tv[:,:,:,1:] = (x[:,:,:,1:]-x[:,:,:,:-1])**2
        tensor_size = x.numel()
        
        return (h_tv + w_tv).sum()/batch_size
        #return (h_tv/count_h+w_tv/count_w)/batch_size

