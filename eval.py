import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataloader import BinaryLoader
from skimage import measure, morphology
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import cv2
import os
from skimage import io, transform
from PIL import Image
import json
from tqdm import tqdm
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from model import MHRMedSeg
from functools import partial
from scipy import ndimage as ndi
from monai.metrics import compute_hausdorff_distance, compute_percent_hausdorff_distance, HausdorffDistanceMetric


# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
torch.set_num_threads(4)

def hd_score(p, y):

    tmp_hd = compute_hausdorff_distance(p, y)
    tmp_hd = torch.mean(tmp_hd)

    return tmp_hd.item()

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='isic',type=str, help='isic, cvccdb, lung, colon, dsb, ultrasound')
    parser.add_argument('--jsonfile', default='data_split.json',type=str, help='')
    parser.add_argument('--mode', default='2d',type=str, help='2d, 3d')
    parser.add_argument('--gt_path', default='mask_1024',type=str, help='mask_1024_c1, mask_1024_c2')
    parser.add_argument('--model',default='model.pth', type=str, help='the path of model')
    parser.add_argument('--dim', type=int, default=96, help='epoches')
    parser.add_argument('--size', type=int, default=1024, help='epoches')
    args = parser.parse_args()

    pred_path = f'visual/{args.dataset}/{args.model}/'

    os.makedirs(pred_path, exist_ok=True)

    print(args.dataset)
    print("------------------------------------------")

    if args.mode == "2d":
        args.jsonfile = f'/home/****/hd1/****/datasets/{args.dataset}_data_split.json'
        if args.dataset == "refuge":
            args.jsonfile = f'/home/****/datasets/REFUGE/data_split.json'
    
        with open(args.jsonfile, 'r') as f:
            df = json.load(f)

        test_files = df['test'] #+ df['train'] + df['valid']

    else:

        args.jsonfile = f'/home/****/datasets/Synapse/test_data_split.json'
        jsonfile_train = f'/home/****/datasets/Synapse/train_data_split.json'
        
        with open(args.jsonfile, 'r') as f:
            df = json.load(f)
        
        with open(jsonfile_train, 'r') as f:
            df2 = json.load(f)

        test_files = df2[args.dataset] + df[args.dataset] # aorta gb liver lk pc rk sm sp

    test_dataset = BinaryLoader(args.gt_path, test_files, A.Compose([
                                        A.Resize(1024, 1024),
                                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ToTensor()
                                        ]))

    model = MHRMedSeg(dim=args.dim, img_size=args.size)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)
    
    model.load_state_dict(torch.load(f'outputs/{args.model}'), strict=True)



    # pretrain_dict = torch.load(f'outputs/{args.model}')
    # selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'image_encoder'}
    # neck_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'neck'}
    # model.image_encoder.load_state_dict(selected_dict, strict=True)
    # model.neck.load_state_dict(neck_dict, strict=True)

    # sam_dict = torch.load('/home/****/pretrain/sam_vit_b_01ec64.pth')

    # selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in sam_dict.items() if list(k.split('.'))[0] == 'prompt_encoder'}
    # model.prompt_encoder.load_state_dict(selected_dict, strict=True)

    # selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in sam_dict.items() if list(k.split('.'))[0] == 'mask_decoder'}
    # model.mask_decoder.load_state_dict(selected_dict, strict=True)

    model = model.cuda()

    model.train(False)  
    
    TestAcc = Accuracy()
    TestPrecision = Precision()
    TestDice = Dice()
    TestRecall = Recall()
    TestF1 = F1(2)
    TestIoU = IoU()

    mIoU = []
    Acc = []
    Prec = []
    Rec = []
    F1_score = []
    DSC = []
    FPS = []
    image_ids = []
    hd_list = []
    
    since = time.time()
    
    with torch.no_grad():
        for _, img, mask, img_id in tqdm(test_dataset):
        # for _, img, mask, img_id in test_dataset:

            img = Variable(torch.unsqueeze(img, dim=0), requires_grad=False).cuda()            
            mask = Variable(torch.unsqueeze(mask, dim=0), requires_grad=False).cuda()

            torch.cuda.synchronize()
            start = time.time()

            mask_pred = model(x=img, mask=mask)
            # print(mask_pred.shape)

            torch.cuda.synchronize()
            end = time.time()
            FPS.append(end-start)

            mask_pred = torch.sigmoid(mask_pred)

            mask_pred[mask_pred >= 0.5] = 1
            mask_pred[mask_pred < 0.5] = 0

            # for box prompt
            if len(mask_pred.shape) > 4:
                mask_pred = torch.sum(mask_pred,dim=1, keepdim=True)#.squeeze(1)
                mask_pred = torch.squeeze(mask_pred, dim=1)
                mask_pred = torch.sum(mask_pred,dim=1, keepdim=True)

            # print(mask_pred.shape)


            mask_draw = mask_pred.clone().detach()
            gt_draw = mask.clone().detach()
            

            iou = TestIoU(mask_pred,mask)
            dsc = TestDice(mask_pred,mask)
            hdscore = hd_score(mask_pred,mask)

            mask_pred = mask_pred.view(-1)
            mask = mask.view(-1)


            img_id = list(img_id.split('.'))[0]
            mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
            mask_numpy[mask_numpy==1] = 255 
            # cv2.imwrite(f'{pred_path}{img_id}.png',mask_numpy)

     
            accuracy = TestAcc(mask_pred.cpu(),mask.cpu())
            precision = TestPrecision(mask_pred.cpu(),mask.cpu())
            recall = TestRecall(mask_pred.cpu(),mask.cpu())
            f1score = TestF1(mask_pred.cpu(),mask.cpu())

         
            mIoU.append(iou.item())
            DSC.append(dsc.item())
            Acc.append(accuracy.item())
            Prec.append(precision.item())
            Rec.append(recall.item())
            F1_score.append(f1score.item())
            image_ids.append(img_id)
            if hdscore != float("inf"):
                hd_list.append(hdscore)
            torch.cuda.empty_cache()

    time_elapsed = time.time() - since

    result_dict = {'image_id':image_ids, 'miou':mIoU, 'dice':DSC}
    result_df = pd.DataFrame(result_dict)
    # result_df.to_csv(f'visual/{args.dataset}/results_v2_{args.dataset}.csv',index=False)
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    F1 = 2 * np.mean(Prec) * np.mean(Rec) / (np.mean(Prec) + np.mean(Rec))
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('FPS: {:.2f}'.format(1.0/(sum(FPS)/len(FPS))))
    print('mean IoU:',round(np.mean(mIoU),4),round(np.std(mIoU),4))
    print('mean accuracy:',round(np.mean(Acc),4),round(np.std(Acc),4))
    print('mean Prec:',round(np.mean(Prec),4))
    print('mean Rec:',round(np.mean(Rec),4))
    print('mean F1:',round(np.mean(F1),4))
    print('mean HD:',round(np.mean(hd_list),4),round(np.std(hd_list),4))
    print('mean Dice:',round(np.mean(DSC),4),round(np.std(DSC),4))

