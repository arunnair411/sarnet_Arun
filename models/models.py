import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from .model_parts import up_1d, down_1d, outconv_1d, down_separable, up_separable, outconv_separable

class UNetSAR(nn.Module):
    def __init__(self):
        super(UNetSAR,self).__init__()
        self.enc_conv1 = nn.Conv1d(1,64, kernel_size=4, stride=2)
        self.enc_conv2 = nn.Conv1d(64,128, kernel_size=4, stride=2)
        self.enc_conv3 = nn.Conv1d(128,256, kernel_size=4, stride=2)
        self.enc_conv4 = nn.Conv1d(256,512, kernel_size=4, stride=2)
        self.enc_conv5 = nn.Conv1d(512,512, kernel_size=4, stride=2)
        self.enc_conv6 = nn.Conv1d(512,512, kernel_size=4, stride=2)
        self.enc_conv7 = nn.Conv1d(512,512, kernel_size=4, stride=2)
        self.enc_conv8 = nn.Conv1d(512,512, kernel_size=4, stride=2)

        self.enc_bn1 = nn.BatchNorm1d(64)
        self.enc_bn2 = nn.BatchNorm1d(128)
        self.enc_bn3 = nn.BatchNorm1d(256)
        self.enc_bn4 = nn.BatchNorm1d(512)
        self.enc_bn5 = nn.BatchNorm1d(512)
        self.enc_bn6 = nn.BatchNorm1d(512)
        self.enc_bn7 = nn.BatchNorm1d(512)
        self.enc_bn8 = nn.BatchNorm1d(512)

        self.dec_conv8 = nn.ConvTranspose1d(512,512, kernel_size=4, stride=2, output_padding=1)
        self.dec_conv7 = nn.ConvTranspose1d(1024,512, kernel_size=4, stride=2, output_padding=1)
        self.dec_conv6 = nn.ConvTranspose1d(1024,512, kernel_size=4, stride=2, output_padding=1)
        self.dec_conv5 = nn.ConvTranspose1d(1024,512, kernel_size=4, stride=2)
        self.dec_conv4 = nn.ConvTranspose1d(1024,256, kernel_size=4, stride=2, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose1d(512,128, kernel_size=4, stride=2)
        self.dec_conv2 = nn.ConvTranspose1d(256,64, kernel_size=4, stride=2, output_padding=1)
        self.dec_conv1 = nn.ConvTranspose1d(128,1, kernel_size=4, stride=2)

        self.dec_bn2 = nn.BatchNorm1d(64)
        self.dec_bn3 = nn.BatchNorm1d(128)
        self.dec_bn4 = nn.BatchNorm1d(256)
        self.dec_bn5 = nn.BatchNorm1d(512)
        self.dec_bn6 = nn.BatchNorm1d(512)
        self.dec_bn7 = nn.BatchNorm1d(512)
        self.dec_bn8 = nn.BatchNorm1d(512)

        self.fc = nn.Linear(1000,1000)

    # @autocast()
    def forward(self, x):
        with autocast(enabled=False): # Alternatively...
            x1 = F.leaky_relu(self.enc_bn1(self.enc_conv1(x)), negative_slope=0.2)
            x2 = F.leaky_relu(self.enc_bn2(self.enc_conv2(x1)), negative_slope=0.2)
            x3 = F.leaky_relu(self.enc_bn3(self.enc_conv3(x2)), negative_slope=0.2)
            x4 = F.leaky_relu(self.enc_bn4(self.enc_conv4(x3)), negative_slope=0.2)
            x5 = F.leaky_relu(self.enc_bn5(self.enc_conv5(x4)), negative_slope=0.2)
            x6 = F.leaky_relu(self.enc_bn6(self.enc_conv6(x5)), negative_slope=0.2)
            x7 = F.leaky_relu(self.enc_bn7(self.enc_conv7(x6)), negative_slope=0.2)
            # x8 = F.leaky_relu(self.enc_bn8(self.enc_conv8(x7)), negative_slope=0.2)
            x8 = F.leaky_relu(self.enc_conv8(x7), negative_slope=0.2)

            xd1 = F.leaky_relu(self.dec_bn8(self.dec_conv8(x8)), negative_slope=0.2)
            xd2 = F.leaky_relu(self.dec_bn7(self.dec_conv7(torch.cat([xd1,x7], dim=1))), negative_slope=0.2)
            xd3 = F.leaky_relu(self.dec_bn6(self.dec_conv6(torch.cat([xd2,x6], dim=1))), negative_slope=0.2)
            xd4 = F.leaky_relu(self.dec_bn5(self.dec_conv5(torch.cat([xd3,x5], dim=1))), negative_slope=0.2)
            xd5 = F.leaky_relu(self.dec_bn4(self.dec_conv4(torch.cat([xd4,x4], dim=1))), negative_slope=0.2)
            xd6 = F.leaky_relu(self.dec_bn3(self.dec_conv3(torch.cat([xd5,x3], dim=1))), negative_slope=0.2)
            xd7 = F.leaky_relu(self.dec_bn2(self.dec_conv2(torch.cat([xd6,x2], dim=1))), negative_slope=0.2)
            xd8 = self.dec_conv1(torch.cat([xd7,x1], dim=1))
            x_out = self.fc(xd8)
            return x_out

# Akshay's net is a little wonky... has edge effects and stuff he unevenly compensates for. Instead doing the below.
class UNetSAR_Arun(nn.Module): # One single encoder branch, one decoder branch
    def __init__(self, in_chans=1, out_chans=1, normalization='batchnorm'):
        super(UNetSAR_Arun, self).__init__()
        # self.dropout_rate = dropout_rate
        # self.l2_reg = l2_reg
        
        # Encoders - downsampling branch
        # Inpus is: input_data.shape = batch_size x in_chans x 16384 (B x C x T=TimeSamples)
        kernel_size = (5,)
        stride = (2,)
        stridex2 = (4,) # Twice the stride
        padding = ( int((kernel_size[0]-1)/2), )
        self.e00 = down_1d(in_chans=in_chans, out_chans=64,  kernel_size=kernel_size, stride=(1,),   padding=padding, _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x   64 x 1024 
        self.e01 = down_1d(in_chans=64,       out_chans=128, kernel_size=kernel_size, stride=stride, padding=padding, _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x  128 x  512
        self.e02 = down_1d(in_chans=128,      out_chans=128, kernel_size=kernel_size, stride=stride, padding=padding, _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x  128 x  256
        self.e03 = down_1d(in_chans=128,      out_chans=128, kernel_size=kernel_size, stride=stride, padding=padding, _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x  128 x  128        
        self.e04 = down_1d(in_chans=128,      out_chans=128, kernel_size=kernel_size, stride=stride, padding=padding, _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x  128 x   64
        self.e05 = down_1d(in_chans=128,      out_chans=256, kernel_size=kernel_size, stride=stride, padding=padding, _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x  256 x   32
        self.e06 = down_1d(in_chans=256,      out_chans=256, kernel_size=kernel_size, stride=stride, padding=padding, _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x  256 x   16
        self.e07 = down_1d(in_chans=256,      out_chans=256, kernel_size=kernel_size, stride=stride, padding=padding, _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x  256 x    8
        self.e08 = down_1d(in_chans=256,      out_chans=256, kernel_size=kernel_size, stride=stride, padding=padding, _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x  256 x    4
        self.e09 = down_1d(in_chans=256,      out_chans=512, kernel_size=(4,),        stride=stride, padding=(0,),    _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x  512 x    1

        # Decoders - upsampling branch
        # Inpus is: input_data.shape = batch_size x 512 x   1 (B x C x T=TimeSamples)
        self.d09 = up_1d     (in_chans=512,      out_chans=256, kernel_size=(1,),        stride=(1,), padding=(0,),    _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=stridex2, normalization=normalization) # Output is batch_size x 256 x    4, then concatenated with e08 - batch_size x 256 x    4
        self.d08 = up_1d     (in_chans=512,      out_chans=256, kernel_size=(1,),        stride=(1,), padding=(0,),    _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=stride, normalization=normalization)   # Output is batch_size x 256 x    8, then concatenated with e07 - batch_size x 256 x    8
        self.d07 = up_1d     (in_chans=512,      out_chans=256, kernel_size=kernel_size, stride=(1,), padding=padding, _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=stride, normalization=normalization)   # Output is batch_size x 256 x   16, then concatenated with e06 - batch_size x 256 x   16
        self.d06 = up_1d     (in_chans=512,      out_chans=256, kernel_size=kernel_size, stride=(1,), padding=padding, _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=stride, normalization=normalization)   # Output is batch_size x 256 x   32, then concatenated with e05 - batch_size x 256 x   32
        self.d05 = up_1d     (in_chans=512,      out_chans=128, kernel_size=kernel_size, stride=(1,), padding=padding, _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=stride, normalization=normalization)   # Output is batch_size x 128 x   64, then concatenated with e04 - batch_size x 128 x   64
        self.d04 = up_1d     (in_chans=256,      out_chans=128, kernel_size=kernel_size, stride=(1,), padding=padding, _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=stride, normalization=normalization)   # Output is batch_size x 128 x  128, then concatenated with e03 - batch_size x 128 x  128
        self.d03 = up_1d     (in_chans=256,      out_chans=128, kernel_size=kernel_size, stride=(1,), padding=padding, _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=stride, normalization=normalization)   # Output is batch_size x 128 x  256, then concatenated with e02 - batch_size x 128 x  256
        self.d02 = up_1d     (in_chans=256,      out_chans=128, kernel_size=kernel_size, stride=(1,), padding=padding, _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=stride, normalization=normalization)   # Output is batch_size x 128 x  512, then concatenated with e01 - batch_size x 128 x  512
        self.d01 = up_1d     (in_chans=256,      out_chans=64,  kernel_size=kernel_size, stride=(1,), padding=padding, _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=stride, normalization=normalization)   # Output is batch_size x  64 x 1024, then concatenated with e00 - batch_size x  64 x 1024

        self.d00 = outconv_1d(in_chans=128,      out_chans=1,   kernel_size=kernel_size, stride=(1,), padding=padding) # Output is batch_size x 1 x 1024

    def forward(self, x):
        with autocast(enabled=False): # Alternatively...
            e00 = self.e00(x)
            e01 = self.e01(e00)
            e02 = self.e02(e01)
            e03 = self.e03(e02)
            e04 = self.e04(e03)
            e05 = self.e05(e04)
            e06 = self.e06(e05)
            e07 = self.e07(e06)
            e08 = self.e08(e07)
            e09 = self.e09(e08)

            _encoding = e09

            d09 = self.d09(_encoding)
            d08 = self.d08(d09, e08)
            d07 = self.d07(d08, e07)
            d06 = self.d06(d07, e06)
            d05 = self.d05(d06, e05)
            d04 = self.d04(d05, e04)
            d03 = self.d03(d04, e03)
            d02 = self.d02(d03, e02)
            d01 = self.d01(d02, e01)
            d00 = self.d00(d01, e00)

            predicted_signal = d00
            return predicted_signal

class UNet2DSAR_slowfirst_3(nn.Module): # One single encoder branch, one decoder branch
    def __init__(self, in_chans=1, out_chans=1, normalization='batchnorm'):
        super(UNet2DSAR_slowfirst_3, self).__init__()
        # self.dropout_rate = dropout_rate
        # self.l2_reg = l2_reg
        
        # Encoders - downsampling branch
        # Inpus is: input_data.shape = batch_size x in_chans x 3/5 x 1024 (B x C x H=#SlowTime x W=#FastTime)
        # e00 = tf.transpose(LB, perm=[0,2,1,3]) # Input is originally 256x64, but uros makes it 64x256 through this statement
        self.e01 = down_separable(in_chans=in_chans, out_chans=64,  kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x  64 x 3 x 1024
        self.e02 = down_separable(in_chans=64,       out_chans=128, kernel_size=(3, 5), stride=(2, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 128 x 1 x 1024
        self.e03 = down_separable(in_chans=128,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x  512
        self.e04 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x  256
        self.e05 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x  128
        self.e06 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x   64
        self.e07 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x   32
        self.e08 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x   16
        self.e09 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x    8
        self.e10 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x    4
        self.e11 = down_separable(in_chans=256,      out_chans=512, kernel_size=(1, 4), stride=(1, 1), padding=(0, 0), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 512 x 1 x    1 

        # Decoders - upsampling branch
        # Inpus is: input_data.shape = batch_size x 256 x   1 x  1 (B x C x H=Freqs x W=TimeFrames)
        self.d11 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,4), normalization=normalization, dropout_p=0.0) # Output is batch_size x 256 x   1 x  4, then concatenated with e10 - batch_size x 256 x   1 x  4
        self.d10 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x    8, then concatenated with e09 - batch_size x 256 x   1 x    8
        self.d09 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x   16, then concatenated with e08 - batch_size x 256 x   1 x   16
        self.d08 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x   32, then concatenated with e07 - batch_size x 256 x   1 x   32
        self.d07 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x   64, then concatenated with e06 - batch_size x 256 x   1 x   64
        self.d06 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x  128, then concatenated with e05 - batch_size x 256 x   1 x  128
        self.d05 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x  256, then concatenated with e04 - batch_size x 256 x   1 x  256
        self.d04 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x  512, then concatenated with e03 - batch_size x 256 x   1 x  512
        self.d03 = up_separable(in_chans=512,      out_chans=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 128 x   1 x 1024, then concatenated with e02 - batch_size x 128 x   1 x 1024
        self.d02 = up_separable(in_chans=256,      out_chans= 64, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(3,1), normalization=normalization) # Output is batch_size x  64 x   3 x 1024, then concatenated with e01 - batch_size x  64 x   3 x 1024

        self.d01 = outconv_separable(in_chans=128,   out_chans=out_chans,   kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)) # Output is batch_size x   1 x 3 x 1024

    def forward(self, x):
        e01 = self.e01(x)
        e02 = self.e02(e01)
        e03 = self.e03(e02)
        e04 = self.e04(e03)
        e05 = self.e05(e04)
        e06 = self.e06(e05)
        e07 = self.e07(e06)
        e08 = self.e08(e07)
        e09 = self.e09(e08)
        e10 = self.e10(e09)
        e11 = self.e11(e10)

        _encoding = e11

        d11 = self.d11(_encoding)
        d10 = self.d10(d11, e10)
        d09 = self.d09(d10, e09)
        d08 = self.d08(d09, e08)
        d07 = self.d07(d08, e07)
        d06 = self.d06(d07, e06)
        d05 = self.d05(d06, e05)
        d04 = self.d04(d05, e04)
        d03 = self.d03(d04, e03)
        d02 = self.d02(d03, e02)
        d01 = self.d01(d02, e01)

        predicted_signal = d01
        return predicted_signal

class UNet2DSAR_fastfirst_3(nn.Module): # One single encoder branch, one decoder branch
    def __init__(self, in_chans=1, out_chans=1, normalization='batchnorm'):
        super(UNet2DSAR_fastfirst_3, self).__init__()
        # self.dropout_rate = dropout_rate
        # self.l2_reg = l2_reg
        
        # Encoders - downsampling branch
        # Inpus is: input_data.shape = batch_size x in_chans x 3/5 x 1024 (B x C x H=#SlowTime x W=#FastTime)
        # e00 = tf.transpose(LB, perm=[0,2,1,3]) # Input is originally 256x64, but uros makes it 64x256 through this statement
        self.e01 = down_separable(in_chans=in_chans, out_chans=64,  kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x  64 x 3 x 1024
        self.e02 = down_separable(in_chans=64,       out_chans=128, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 128 x 3 x  512
        self.e03 = down_separable(in_chans=128,      out_chans=256, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 3 x  256
        self.e04 = down_separable(in_chans=256,      out_chans=256, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 3 x  128
        self.e05 = down_separable(in_chans=256,      out_chans=256, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 3 x   64
        self.e06 = down_separable(in_chans=256,      out_chans=256, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 3 x   32
        self.e07 = down_separable(in_chans=256,      out_chans=256, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 3 x   16
        self.e08 = down_separable(in_chans=256,      out_chans=256, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 3 x    8
        self.e09 = down_separable(in_chans=256,      out_chans=256, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 3 x    4
        self.e10 = down_separable(in_chans=256,      out_chans=256, kernel_size=(3, 4), stride=(1, 1), padding=(1, 0), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 3 x    1
        self.e11 = down_separable(in_chans=256,      out_chans=512, kernel_size=(3, 1), stride=(1, 1), padding=(0, 0), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 512 x 1 x    1

        # Decoders - upsampling branch
        # Inpus is: input_data.shape = batch_size x 256 x   1 x  1 (B x C x H=Freqs x W=TimeFrames)
        self.d11 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(3,1), normalization=normalization, dropout_p=0.0) # Output is batch_size x 256 x   3 x  1, then concatenated with e10 - batch_size x 256 x   3 x  1
        self.d10 = up_separable(in_chans=512,      out_chans=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,4), normalization=normalization) # Output is batch_size x 256 x   3 x    4, then concatenated with e09 - batch_size x 256 x   3 x    4
        self.d09 = up_separable(in_chans=512,      out_chans=256, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   3 x    8, then concatenated with e08 - batch_size x 256 x   3 x    8
        self.d08 = up_separable(in_chans=512,      out_chans=256, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   3 x   16, then concatenated with e07 - batch_size x 256 x   3 x   16
        self.d07 = up_separable(in_chans=512,      out_chans=256, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   3 x   32, then concatenated with e06 - batch_size x 256 x   3 x   32
        self.d06 = up_separable(in_chans=512,      out_chans=256, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   3 x   64, then concatenated with e05 - batch_size x 256 x   3 x   64
        self.d05 = up_separable(in_chans=512,      out_chans=256, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   3 x  128, then concatenated with e04 - batch_size x 256 x   3 x  128
        self.d04 = up_separable(in_chans=512,      out_chans=256, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   3 x  256, then concatenated with e03 - batch_size x 256 x   3 x  256
        self.d03 = up_separable(in_chans=512,      out_chans=128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 128 x   3 x  512, then concatenated with e02 - batch_size x 128 x   3 x  512
        self.d02 = up_separable(in_chans=256,      out_chans= 64, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x  64 x   3 x 1024, then concatenated with e01 - batch_size x  64 x   3 x 1024

        self.d01 = outconv_separable(in_chans=128,   out_chans=out_chans,   kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)) # Output is batch_size x   1 x 5 x 1024

    def forward(self, x):
        e01 = self.e01(x)
        e02 = self.e02(e01)
        e03 = self.e03(e02)
        e04 = self.e04(e03)
        e05 = self.e05(e04)
        e06 = self.e06(e05)
        e07 = self.e07(e06)
        e08 = self.e08(e07)
        e09 = self.e09(e08)
        e10 = self.e10(e09)
        e11 = self.e11(e10)

        _encoding = e11

        d11 = self.d11(_encoding)
        d10 = self.d10(d11, e10)
        d09 = self.d09(d10, e09)
        d08 = self.d08(d09, e08)
        d07 = self.d07(d08, e07)
        d06 = self.d06(d07, e06)
        d05 = self.d05(d06, e05)
        d04 = self.d04(d05, e04)
        d03 = self.d03(d04, e03)
        d02 = self.d02(d03, e02)

        d01 = self.d01(d02, e01)

        predicted_signal = d01
        return predicted_signal

# TODO: Do 6 instead of 5... pixelshuffling issue... INCOMPLETE
class UNet2DSAR_slowfirst_5(nn.Module): # One single encoder branch, one decoder branch
    def __init__(self, in_chans=1, out_chans=1, normalization='batchnorm'):
        super(UNet2DSAR_slowfirst_5, self).__init__()
        # self.dropout_rate = dropout_rate
        # self.l2_reg = l2_reg
        
        # Encoders - downsampling branch
        # Inpus is: input_data.shape = batch_size x in_chans x 3/5 x 1024 (B x C x H=#SlowTime x W=#FastTime)
        # e00 = tf.transpose(LB, perm=[0,2,1,3]) # Input is originally 256x64, but uros makes it 64x256 through this statement
        # TODO: Try with smaller time kernels - 3 instead of 5
        self.e01 = down_separable(in_chans=in_chans, out_chans=64,  kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x  64 x 5 x 1024
        self.e02 = down_separable(in_chans=64,       out_chans=128, kernel_size=(3, 5), stride=(2, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 128 x 3 x 1024
        self.e03 = down_separable(in_chans=128,      out_chans=128, kernel_size=(3, 5), stride=(2, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 128 x 1 x 1024
        self.e04 = down_separable(in_chans=128,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x  512
        self.e05 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x  256
        self.e06 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x  128
        self.e07 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x   64
        self.e08 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x   32
        self.e09 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x   16
        self.e10 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x    8
        self.e11 = down_separable(in_chans=256,      out_chans=256, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 256 x 1 x    4
        self.e12 = down_separable(in_chans=256,      out_chans=512, kernel_size=(1, 4), stride=(1, 1), padding=(0, 0), _n_momentum=0.9, lrelu_slope=0.2, normalization=normalization) # Output is batch_size x 512 x 1 x    1 

        # Decoders - upsampling branch
        # Inpus is: input_data.shape = batch_size x 256 x   1 x  1 (B x C x H=Freqs x W=TimeFrames)
        self.d12 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,4), normalization=normalization, dropout_p=0.0) # Output is batch_size x 256 x   1 x  4, then concatenated with e11 - batch_size x 256 x   1 x  4
        self.d11 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x    8, then concatenated with e10 - batch_size x 256 x   1 x    8
        self.d10 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x   16, then concatenated with e09 - batch_size x 256 x   1 x   16
        self.d09 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x   32, then concatenated with e08 - batch_size x 256 x   1 x   32
        self.d08 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x   64, then concatenated with e07 - batch_size x 256 x   1 x   64
        self.d07 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x  128, then concatenated with e06 - batch_size x 256 x   1 x  128
        self.d06 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x  256, then concatenated with e05 - batch_size x 256 x   1 x  256
        self.d05 = up_separable(in_chans=512,      out_chans=256, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 256 x   1 x  512, then concatenated with e04 - batch_size x 256 x   1 x  512
        self.d04 = up_separable(in_chans=512,      out_chans=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(1,2), normalization=normalization) # Output is batch_size x 128 x   1 x 1024, then concatenated with e03 - batch_size x 128 x   1 x 1024
        self.d03 = up_separable(in_chans=256,      out_chans=128, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(3,1), normalization=normalization) # Output is batch_size x 128 x   3 x 1024, then concatenated with e02 - batch_size x 128 x   3 x 1024
        self.d02 = up_separable(in_chans=256,      out_chans=64,  kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), _n_momentum=0.9, lrelu_slope=0.2, pixelshuffler_strides=(2,1), normalization=normalization) # Output is batch_size x  64 x *6* x 1024, then concatenated with e01 - batch_size x  64 x   5 x 1024 - need to discard one dimension....
        self.d01 = outconv_separable(in_chans=128,   out_chans=out_chans,   kernel_size=(3, 5), stride=(1, 1), padding=(1, 2)) # Output is batch_size x   1 x 5 x 1024

    def forward(self, x):
        e01 = self.e01(x)
        e02 = self.e02(e01)
        e03 = self.e03(e02)
        e04 = self.e04(e03)
        e05 = self.e05(e04)
        e06 = self.e06(e05)
        e07 = self.e07(e06)
        e08 = self.e08(e07)
        e09 = self.e09(e08)
        e10 = self.e10(e09)
        e11 = self.e11(e10)
        e12 = self.e12(e11)
        e13 = self.e13(e12)
        e14 = self.e14(e13)
        e15 = self.e15(e14)

        speech_encoding = e15

        d15 = self.d15(speech_encoding)
        d14 = self.d14(d15, e14)
        d13 = self.d13(d14, e13)
        d12 = self.d12(d13, e12)
        d11 = self.d11(d12, e11)
        d10 = self.d10(d11, e10)
        d09 = self.d09(d10, e09)
        d08 = self.d08(d09, e08)
        d07 = self.d07(d08, e07)
        d06 = self.d06(d07, e06)
        d05 = self.d05(d06, e05)
        d04 = self.d04(d05, e04)
        d03 = self.d03(d04, e03)
        d02 = self.d02(d03, e02)

        d01 = self.d01(d02, e01)

        predicted_speech = d01
        return predicted_speech