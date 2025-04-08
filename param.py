from self_attention_cv.transunet import TransUnet
from model import MHRMedSeg, PFD, PFDTeacher
from thop import profile
import torch

# model = unet.Model()
model = MHRMedSeg(dim=96, img_size=1024)


randn_input = torch.randn(1, 3, 1024, 1024)
flops, params = profile(model, inputs=(randn_input, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')

total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total_num/1000**2, trainable_num/1000**2)
