import torch
import torch.nn as nn


class MaxPool(nn.Module):
    def __init__(self, kernel_size=2, dim=3, return_indieces=False):
        super(MaxPool, self).__init__()
        if dim == 1:
            max_pool = nn.MaxPool1d
        elif dim == 2:
            max_pool = nn.MaxPool2d
        elif dim == 3:
            max_pool = nn.MaxPool3d
        else:
            raise ValueError("Dimension error")
        self.max_pool = max_pool(kernel_size, stride=2, return_indices=return_indieces)
        return

    def forward(self, x):
        return self.max_pool(x)


class MaxUnpool(nn.Module):
    def __init__(self, kernel_size=2, dim=3):
        super(MaxUnpool, self).__init__()
        if dim == 1:
            max_unpool = nn.MaxUnpool1d
        elif dim == 2:
            max_unpool = nn.MaxUnpool2d
        elif dim == 3:
            max_unpool = nn.MaxUnpool3d
        else:
            raise ValueError("Dimension Error")

        self.max_unpool = max_unpool(kernel_size=kernel_size, stride=2)
        return

    def forward(self, x, indices):
        return self.max_unpool(x, indices)


class ConBnRelDp(nn.Module):
    # conv + batch_normalize + relu + dropout
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, activate_unit='relu', same_padding=True,
                 use_bn=False, use_dp=False, p=0.2, reverse=False, group=1, dilation=1, dim=3):
        super(ConBnRelDp, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if dim == 1:
            conv = nn.Conv1d
            batch_norm = nn.BatchNorm1d
            drop_out = nn.Dropout
            convT = nn.ConvTranspose1d
        elif dim == 2:
            conv = nn.Conv2d
            batch_norm = nn.BatchNorm2d
            drop_out = nn.Dropout2d
            convT = nn.ConvTranspose2d
        elif dim == 3:
            conv = nn.Conv3d
            batch_norm = nn.BatchNorm3d
            drop_out = nn.Dropout3d
            convT = nn.ConvTranspose3d
        else:
            raise ValueError("Dimension can only be 1, 2 or 3.")
        if not reverse:
            self.conv = conv(in_ch, out_ch, kernel_size, stride, padding, groups=group, dilation=dilation)
        else:
            self.conv = convT(in_ch, out_ch, kernel_size, stride, padding, groups=group, dilation=dilation)

        self.batch_norm = batch_norm(out_ch) if use_bn else False
        if activate_unit == 'relu':
            self.activate_unit = nn.ReLU(inplace=True)
        elif activate_unit == 'elu':
            self.activate_unit = nn.ELU(inplace=True)
        elif activate_unit == 'leaky_relu':
            self.activate_unit = nn.LeakyReLU(inplace=True)
        elif activate_unit == 'prelu':
            self.activate_unit = nn.PReLU(init=0.01)
        elif activate_unit == 'sigmoid':
            self.activate_unit = nn.Sigmoid()
        else:
            self.activate_unit = False
        self.drop_out = drop_out(p) if use_dp else False

        return

    def forward(self, x):
        x = self.conv(x)
        if self.activate_unit is not False:
            x = self.activate_unit(x)
        if self.batch_norm is not False:
            x = self.batch_norm(x)
        if self.drop_out is not False:
            x = self.drop_out(x)
        return x


class fc_rel(nn.Module):
    def __init__(self, in_ft, out_ft, activate_unit='relu', use_dp=False):
        super(fc_rel, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft)
        if activate_unit == 'relu':
            self.activate_unit = nn.ReLU(inplace=True)
        elif activate_unit == 'elu':
            self.activate_unit == nn.ELU(inplace=True)
        elif activate_unit == 'leaky_relu':
            self.activate_unit == nn.LeakyReLU(inplace=True)
        else:
            self.activate_unit = False
        self.drop_out = nn.Dropout(0.2) if use_dp else False
        return

    def forward(self, x):
        x = self.fc(x)
        if self.activate_unit is not False:
            x = self.activate_unit(x)
        if self.drop_out is not False:
            x = self.drop_out(x)
        return x


class unet_forward_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dim=3, use_bn=False, use_dp=False):
        super(unet_forward_conv, self).__init__()
        if dim == 1:
            conv = nn.Conv1d
            batch_norm = nn.BatchNorm1d
            drop_out = nn.Dropout
        elif dim == 2:
            conv = nn.Conv2d
            batch_norm = nn.BatchNorm2d
            drop_out = nn.Dropout2d
        elif dim == 3:
            conv = nn.Conv3d
            batch_norm = nn.BatchNorm3d
            drop_out = nn.Dropout3d
        else:
            raise ValueError("Dimension can only be 1, 2 or 3.")

        layer = []
        layer.append(conv(in_ch, out_ch, kernel_size=3, padding=1))
        if use_bn is not False:
            layer.append(batch_norm(out_ch))
        layer.append(nn.LeakyReLU(inplace=True))
        if use_dp is not False:
            layer.append(drop_out(0.2))

        layer.append(conv(out_ch, out_ch, kernel_size=3, padding=1))
        if use_bn is not False:
            layer.append(batch_norm(out_ch))
        layer.append(nn.LeakyReLU(inplace=True))
        if use_dp is not False:
            layer.append(drop_out(0.2))

        self.conv = nn.Sequential(*layer)

    def forward(self, x):
        return self.conv(x)


class unet_down_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dim=3, use_bn=False, use_dp=False):
        super(unet_down_conv, self).__init__()
        if dim == 1:
            mp = nn.MaxPool1d
        elif dim == 2:
            mp = nn.MaxPool2d
        elif dim == 3:
            mp = nn.MaxPool3d
        else:
            raise ValueError("Dimension can only be 1, 2 or 3.")

        self.conv = nn.Sequential(
            mp(2), unet_forward_conv(in_ch, out_ch, dim, use_bn, use_dp)
        )

    def forward(self, x):
        return self.conv(x)


class unet_up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dim=3, use_bn=False, use_dp=False):
        super(unet_up_conv, self).__init__()
        if dim == 1:
            conv_t = nn.ConvTranspose1d
        elif dim == 2:
            conv_t = nn.ConvTranspose2d
        elif dim == 3:
            conv_t = nn.ConvTranspose3d
        else:
            raise ValueError("Dimension can only be 1, 2 or 3.")

        self.up = conv_t(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = unet_forward_conv(in_ch, out_ch, dim, use_bn, use_dp)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        assert (x1.size() == x2.size())
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class unet_out_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dim=3):
        super(unet_out_conv, self).__init__()
        if dim == 1:
            conv = nn.Conv1d
        elif dim == 2:
            conv = nn.Conv2d
        elif dim == 3:
            conv = nn.Conv3d
        else:
            raise ValueError("Dimension can only be 1, 2 or 3")

        self.conv = conv(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x
