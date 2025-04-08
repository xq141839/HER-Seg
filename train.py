import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from dataloader import BinaryLoader
from loss import *
from tqdm import tqdm
import json
from model import MHRMedSeg
from functools import partial
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from torchmetrics.classification import BinaryAccuracy
import wandb

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
torch.set_num_threads(4)


def train_model(model, optimizer, scheduler, num_epochs=5):
    since = time.time()
    
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    
    best_model_wts = model.state_dict()

    best_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
    
            else:
                model.train(False)  

            running_loss = []
            running_corrects_mask = []
            # running_loss_patch = []
            # running_corrects_patch = []
        
            # Iterate over data
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            for img_res, img, labels, img_id in tqdm(dataloaders[phase]):      
                # wrap them in Variable
                if torch.cuda.is_available():
    
                    img = Variable(img_res.cuda())
                    labels = Variable(labels.cuda())
                 
                else:
                    img, labels = Variable(img), Variable(labels)

                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                pred_mask = model(x=img, mask=labels)

                pred_mask = torch.sigmoid(pred_mask)

                score_mask = iou_metric(pred_mask, labels)



                loss = mask_loss(pred_mask, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # calculate loss and IoU
                running_loss.append(loss.item())
                running_corrects_mask.append(score_mask.item())
             

            epoch_loss = np.mean(running_loss)
            epoch_mask_iou = np.mean(running_corrects_mask)

            # if phase == 'train':
            #     wandb.log({"train-IoU": epoch_mask_iou, "train-loss": epoch_loss})
            # else:
            #     wandb.log({"valid-IoU": epoch_mask_iou, "valid-loss": epoch_loss})
            
            print('{} Loss: {:.4f} Mask IoU: {:.4f}'.format(
                phase, np.mean(epoch_loss), np.mean(epoch_mask_iou)))
            
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_mask_iou)

            # save parameters
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, f'outputs/modelvab_{args.dataset}_epoch_{epoch}.pth')
                counter = 0
            elif phase == 'valid' and epoch_loss > best_loss:
                counter += 1
            if phase == 'valid':
                scheduler.step()
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    


    return Loss_list, Accuracy_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='isic', help='dataset name')
    parser.add_argument('--pretrain', type=str,default='outputs/model_PFD_10K.pth', help='pretraining model')
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='epoches')
    parser.add_argument('--dim', type=int, default=96, help='epoches')
    parser.add_argument('--size', type=int, default=1024, help='epoches')
    parser.add_argument('--ratio', type=float, default=1.0, help='epoches')
    args = parser.parse_args()

    os.makedirs(f'outputs/',exist_ok=True)

    args.jsonfile = f'/home/****/hd1/****/datasets/{args.dataset}_data_split.json'
    
    with open(args.jsonfile, 'r') as f:
        df = json.load(f)

    val_files = df['valid']
    train_files = df['train']

    
    train_dataset = BinaryLoader('mask_1024', train_files, A.Compose([
        A.Resize(args.size, args.size),
        A.HorizontalFlip(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ], 
        additional_targets={'mask2': 'mask'}))
    val_dataset = BinaryLoader('mask_1024', val_files, A.Compose([
        A.Resize(args.size, args.size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ],
        additional_targets={'mask2': 'mask'}))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    
    dataloaders = {'train':train_loader,'valid':val_loader}
    

    model = MHRMedSeg(dim=args.dim, img_size=args.size)

    # model.load_state_dict(torch.load(args.pretrain), strict=True)

    # pretrain_dict = torch.load(args.pretrain)
    # selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'image_encoder'}
    # neck_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'neck'}
    # model.image_encoder.load_state_dict(selected_dict, strict=True)
    # model.neck.load_state_dict(neck_dict, strict=True)

    # sam_dict = torch.load(args.sam_pretrain)

    # selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in sam_dict.items() if list(k.split('.'))[0] == 'prompt_encoder'}
    # model.prompt_encoder.load_state_dict(selected_dict, strict=True)

    # selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in sam_dict.items() if list(k.split('.'))[0] == 'mask_decoder'}
    # model.mask_decoder.load_state_dict(selected_dict, strict=True)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    model = model.cuda()

    # for n, value in model.image_encoder.named_parameters():
    #     value.requires_grad = False

    # for n, value in model.neck.named_parameters():
    #     value.requires_grad = False
    
    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
    )

    total_params = sum(
	param.numel() for param in model.parameters()
    )

    # encoder_params = sum(
	# param.numel() for param in model.image_encoder.parameters()
    # )

    # decoder_params = sum(
	# param.numel() for param in model.mask_decoder.parameters()
    # )

    # print('Encoder Params = ' + str(encoder_params/1000**2) + 'M')
    # print('Decoder Params = ' + str(decoder_params/1000**2) + 'M')

    print('Total Params = ' + str(total_params/1000**2) + 'M')

    print('Ratio = ' + str(trainable_params/total_params) + '%')
    # Loss, IoU and Optimizer
    # kd_loss = nn.MSELoss()
    mask_loss = BinaryMaskLoss(0.8) # nn.CrossEntropyLoss()
    # patch_loss = nn.BCELoss() #BinaryBoundaryLoss()
    iou_metric = BinaryIoU()
    acc_metric = BinaryAccuracy(multidim_average='samplewise')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = args.lr)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98, verbose=True)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5,min_lr=1e-7)
    Loss_list, Accuracy_list = train_model(model, optimizer, exp_lr_scheduler,
                           num_epochs=args.epoch)
    
    plt.title('Validation loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["valid"], 'IoU':Accuracy_list["valid"]})
    valid_data.to_csv(f'valid_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('valid.png')
    
    plt.figure()
    plt.title('Training loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["train"],'IoU':Accuracy_list["train"]})
    valid_data.to_csv(f'train_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('train.png')