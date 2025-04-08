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
from dataloader import KDLoader
from loss import *
from tqdm import tqdm
import json
from model import PFD, PFDTeacher
from functools import partial
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from torchmetrics.classification import BinaryAccuracy

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
torch.set_num_threads(8)


def train_model(model, sam_model, optimizer, scheduler, num_epochs=5):
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
            sam_model.train(False)

            if phase == 'train':
                model.train(True)
    
            else:
                model.train(False)  

            running_loss = []
        
            # Iterate over data
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            for _, img, img_id in tqdm(dataloaders[phase]):      
                # wrap them in Variable
    
                img = Variable(img.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()
                
                student_emb = model(x=img)
                with torch.no_grad():
                    teacher_emb = sam_model(x=img)

                loss = kd_loss(student_emb, teacher_emb)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # calculate loss and IoU
                running_loss.append(loss.item())
             
            epoch_loss = np.mean(running_loss)

            print('{} MSE Loss: {:.4f}'.format(
                phase, np.mean(epoch_loss)))
            
            Loss_list[phase].append(epoch_loss)

            # save parameters
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, f'outputs/modelv3_h_{epoch}.pth')
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
    parser.add_argument('--dataset', type=str,default='all', help='dsb, isic, cvccdb, retinal, lung, ultrasound')
    parser.add_argument('--sam_pretrain', type=str,default='/home/****/pretrain/sam_vit_h_4b8939.pth', help='/home/****/pretrain/mobile_sam.pt, efficient_sam_vitt, sam_vit_b_01ec64, outputs/sam_vit_h_4b8939.pth')
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=20, help='epoches')
    parser.add_argument('--dim', type=int, default=96, help='epoches')
    parser.add_argument('--size', type=int, default=1024, help='epoches')
    args = parser.parse_args()

    os.makedirs(f'outputs/',exist_ok=True)

    args.jsonfile = f'/home/****/datasets/SAM/data_split.json'
    
    with open(args.jsonfile, 'r') as f:
        df = json.load(f)

    val_files = df['valid']
    train_files = df['train']
    
    train_dataset = KDLoader(args.dataset, train_files, A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ]))
    val_dataset = KDLoader(args.dataset, val_files, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ]))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    
    dataloaders = {'train':train_loader,'valid':val_loader}
    

    model = PFD(dim=args.dim, img_size=args.size)
    sam = PFDTeacher()

    pretrain_dict = torch.load(args.sam_pretrain)
    selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'image_encoder'}
    sam.image_encoder.load_state_dict(selected_dict, strict=True)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    model = model.cuda()
    sam = sam.cuda()
        
    # Loss, IoU and Optimizer
    kd_loss = nn.MSELoss()
    # mask_loss = BinaryMaskLoss(0.8) # nn.CrossEntropyLoss()
    # patch_loss = nn.BCELoss() #BinaryBoundaryLoss()

    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98, verbose=True)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5,min_lr=1e-7)
    Loss_list, Accuracy_list = train_model(model, sam, optimizer, exp_lr_scheduler,
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