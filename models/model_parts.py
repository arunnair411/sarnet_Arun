import torch
import torch.nn as nn
import numpy as np

class down_1d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, padding, _n_momentum, lrelu_slope, normalization, include_bias=True):
        super(down_1d, self).__init__()
        normalization_dict = {'batchnorm': nn.BatchNorm1d, 'instancenorm': nn.InstanceNorm1d}
        self.enc = nn.Sequential(
                    nn.Conv1d(in_channels = in_chans, out_channels = out_chans, 
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=include_bias),
                    normalization_dict[normalization](num_features=out_chans, momentum=_n_momentum), # momentum is set to match Dung's code
                    nn.LeakyReLU(negative_slope=lrelu_slope) # negative_slope is set to match Dung's code
        )

    def forward(self, x):
        x = self.enc(x)
        return x

class up_1d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, padding, _n_momentum, lrelu_slope, pixelshuffler_strides, normalization):
        super(up_1d, self).__init__()
        normalization_dict = {'batchnorm': nn.BatchNorm1d, 'instancenorm': nn.InstanceNorm1d}
        self.dec = nn.Sequential(
                    nn.Conv1d(in_channels = in_chans, out_channels = out_chans*np.prod(pixelshuffler_strides), 
                              kernel_size=kernel_size, stride=stride, padding=padding),
                    PixelShuffler1d(shuffle_strides=pixelshuffler_strides, out_filters=out_chans),
                    normalization_dict[normalization](num_features=out_chans, momentum=_n_momentum), # momentum is set to match Dung's code
                    nn.LeakyReLU(negative_slope=lrelu_slope) # negative_slope is set to match Dung's code
        )
    def forward(self, x1, x2 = None):
        if x2 is None:
            x = self.dec(x1)
        elif x2 is not None:
            x = self.dec(torch.cat((x1, x2), 1))
        return x        

class PixelShuffler1d(nn.Module):
    def __init__(self, shuffle_strides=(2,), out_filters=1):
        super(PixelShuffler1d, self).__init__()
        self.shuffle_strides = shuffle_strides
        self.out_filters = out_filters

    def forward(self, inputs):
        batch_size, C, T = inputs.shape
        t, = self.shuffle_strides
        out_c = self.out_filters
        out_t = T * t
        assert C == t * out_c
        
        x = inputs.permute(0, 2, 1)
        x = torch.reshape(x, (batch_size, T, t, out_c))
        x = torch.reshape(x, (batch_size, out_t, out_c))
        x = x.permute(0, 2, 1)
        # x_2 = tf.reshape(inputs, (batch_size, H, W, r1, r2, out_c))
        # x_2 = tf.transpose(x_2, (0, 1, 3, 2, 4, 5))
        # x_2 = tf.reshape(x_2, (batch_size, out_h, out_w, out_c))
        return x

class outconv_1d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, padding):
        super(outconv_1d, self).__init__()
        self.outc_sep = nn.Sequential(
                nn.Conv1d(in_channels = in_chans, out_channels = out_chans,
                    kernel_size=kernel_size, stride=stride, padding=padding)
            )
    def forward(self, x1, x2):
        x = self.outc_sep(torch.cat((x1, x2), 1))
        return x


def truncated_normal_(tensor, mean=0, std=0.02):
    a = mean - 2*std
    b = mean + 2*std
    size = tensor.shape
    tmp = tensor.new_empty(size + (6,)).normal_()
    valid = (tmp < b) & (tmp > a)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

class PixelShuffler(nn.Module):
# https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html - Apparently PyTorch has it implemented! 
# However, since we might want different shuffle_strides in each axis, this is still worth.
# Original paper: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network by Shi et. al (2016)    
    def __init__(self, shuffle_strides=(2, 2), out_filters=1):
        super(PixelShuffler, self).__init__()
        self.shuffle_strides = shuffle_strides
        self.out_filters = out_filters

    def forward(self, inputs):
        batch_size, C, H, W = inputs.shape
        r1, r2 = self.shuffle_strides
        out_c = self.out_filters
        out_h = H * r1
        out_w = W * r2
        assert C == r1 * r2 * out_c
        
        x = inputs.permute(0, 2, 3, 1)
        x = torch.reshape(x, (batch_size, H, W, r1, r2, out_c))
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = torch.reshape(x, (batch_size, out_h, out_w, out_c))
        x = x.permute(0, 3, 1, 2)
        # x_2 = tf.reshape(inputs, (batch_size, H, W, r1, r2, out_c))
        # x_2 = tf.transpose(x_2, (0, 1, 3, 2, 4, 5))
        # x_2 = tf.reshape(x_2, (batch_size, out_h, out_w, out_c))
        return x

class down_separable(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, padding, _n_momentum, lrelu_slope, normalization):
        super(down_separable, self).__init__()
        normalization_dict = {'batchnorm': nn.BatchNorm2d, 'instancenorm': nn.InstanceNorm2d}
        self.enc = nn.Sequential(
                    nn.Conv2d(in_channels = in_chans, out_channels = out_chans, 
                              kernel_size=kernel_size, stride=stride, padding=padding),
                    normalization_dict[normalization](num_features=out_chans, momentum=_n_momentum), # momentum is set to match Dung's code
                    nn.LeakyReLU(negative_slope=lrelu_slope) # negative_slope is set to match Dung's code
        )
        # Set the correct initializations
        truncated_normal_(self.enc[0].weight)   # Initialize weights according to a truncated normal
        nn.init.zeros_(self.enc[0].bias)        # Initialize biases at 0

    def forward(self, x):
        x = self.enc(x)
        return x

class up_separable(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, padding, _n_momentum, lrelu_slope, pixelshuffler_strides, normalization, dropout_p=0):
        super(up_separable, self).__init__()
        normalization_dict = {'batchnorm': nn.BatchNorm2d, 'instancenorm': nn.InstanceNorm2d}
        self.dec = nn.Sequential(
                    nn.Conv2d(in_channels = in_chans, out_channels = out_chans*np.prod(pixelshuffler_strides),
                              kernel_size=kernel_size, stride=stride, padding=padding),
                    PixelShuffler(shuffle_strides=pixelshuffler_strides, out_filters=out_chans),
                    normalization_dict[normalization](num_features=out_chans, momentum=_n_momentum), # momentum is set to match Dung's code
                    nn.Dropout2d(p=dropout_p), # An addition in Uros' code
                    nn.LeakyReLU(negative_slope=lrelu_slope) # negative_slope is set to match Dung's code
        )
        # Set the correct initializations
        truncated_normal_(self.dec[0].weight)   # Initialize weights according to a truncated normal
        nn.init.zeros_(self.dec[0].bias)        # Initialize biases at 0
    def forward(self, x1, x2 = None):
        if x2 is None:
            x = self.dec(x1)
        elif x2 is not None:
            x = self.dec(torch.cat((x1, x2), 1))
        return x

class outconv_separable(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride, padding):
        super(outconv_separable, self).__init__()
        self.outc_sep = nn.Sequential(
            nn.Conv2d(in_channels = in_chans, out_channels = out_chans,
                   kernel_size=kernel_size, stride=stride, padding=padding)
        )
        # Set the correct initializations
        truncated_normal_(self.outc_sep[0].weight)   # Initialize weights according to a truncated normal
        nn.init.zeros_(self.outc_sep[0].bias)        # Initialize biases at 0
    def forward(self, x1, x2):
        x = self.outc_sep(torch.cat((x1, x2),1))
        return x

