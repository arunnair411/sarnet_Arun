import torch
import torch.nn
from torch.autograd import Function, Variable
import math

#################################################################################################################################
# # PSNR Definitions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#################################################################################################################################

# Original psnr formulation defined for the CPU
def psnr(img1, img2):
    """Dice coeff for individual examples"""
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Defining a class to handle PSNR loss evaluation for a batch of data
class PSNRCoeff(Function):
    def __init__(self):
        super(PSNRCoeff,self).__init__()        
    def forward(self, input, target):
        self.mse = torch.mean( (input.view(-1)-target.view(-1))**2 )
        if self.mse == 0:
            return 100
        PIXEL_MAX = 1.0
        t = 20 * math.log10(PIXEL_MAX / math.sqrt(self.mse))
        return t

# Calculating PSNR for a batch of data
def psnr_loss(input, target):
    """PSNR loss for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(input, target)):
        s = s + PSNRCoeff().forward(c[0], c[1])
    return s / (i + 1)

#################################################################################################################################
# # # SNR Definitions # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#################################################################################################################################

# Original snr formulation defined for the CPU
def snr(img1, img2):
    """SNR for img2 (the noisy image) w.r.t. img1 (the clean image)"""

    mse = np.mean( (img1 - img2) ** 2 )
    mss = np.mean( (img1) ** 2 )
    
    if mse == 0:
        return 100

    return 20 * math.log10(math.sqrt(mss) / math.sqrt(mse))

# Defining a class to handle SNR loss evaluation for a batch of data
class SNRCoeff(Function):
    def __init__(self):
        super(SNRCoeff,self).__init__()        
    def forward(self, input, target):
        self.mse = torch.mean( (input.view(-1)-target.view(-1))**2 )
        self.mss = torch.mean( (target.view(-1))**2 )
        if self.mse == 0:
            return 100
        if self.mss == 0:
            return None
        
        t = 10 * torch.log10(self.mss / self.mse)
        return t

# Calculating PSNR for a batch of data
def snr_loss(input, target):
    """SNR loss for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    
    count = 0
    for i, c in enumerate(zip(input, target)):
        return_val = SNRCoeff().forward(c[0], c[1])
        if return_val is None:
            continue
        else:    
            s = s + return_val
            count += 1
    return s / count
