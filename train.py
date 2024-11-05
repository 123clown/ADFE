import time
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import torchvision.transforms as transforms
from dataloader.dataloader import get_DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

from model.swinT import swin_base_patch4_window12_384, swin_base_patch4_window7_224
from model.swinT import load_swin_base_patch4_window12_384, load_swin_base_patch4_window7_224
from model.ADFF import  ADFF

class ADFF_main(object):
    def __init__(self, args):
        self.args = args
        self.epoch = args.epoch
        self.batch_size = args.b
        self.gpu_mode = not args.cpu
        self.seed = args.seed
        self.loss_path = args.loss_path
        self.dump_path = args.dump
        self.test_path = args.t_result
        self.load_dump = (args.load != '')
        self.load_path = Path(args.load)
        self.start_epoch = 1
        self.min_avg = 0.21
        
        
        if not args.test:
            self.set_seed()
            self.train_data_loader, self.test_data_loader = get_DataLoader(args)
            self.loss_path = os.path.join(self.loss_path, time.strftime('%y%m%d-%H%M%S', time.localtime()))
            self.dump_path = os.path.join(self.dump_path, time.strftime('%y%m%d-%H%M%S', time.localtime()))
            self.loss_path = Path(self.loss_path)
            self.dump_path = Path(self.dump_path)
            if not self.loss_path.exists():
                self.loss_path.mkdir()
            if not self.dump_path.exists():
                self.dump_path.mkdir()
        else:
            _, self.test_data_loader = get_DataLoader(args)
        
        if args.size == 224:
            self.swin1 = load_swin_base_patch4_window7_224(args.pre)
            self.swin2 = load_swin_base_patch4_window7_224(args.pre)
        elif args.size == 384:
            self.swin1 = load_swin_base_patch4_window12_384(args.pre)
            self.swin2 = load_swin_base_patch4_window12_384(args.pre)
        
        
        
        self.ADFF = ADFF()
               
        
        
        if args.test:
            for param in self.swin1.parameters():
                param.requires_grad = False
            for param in self.swin2.parameters():
                param.requires_grad = False
            for param in self.C_net.parameters():
                param.requires_grad = False

                    
                    
        # ##           
        self.swin1 = nn.DataParallel(self.swin1)
        self.swin2  = nn.DataParallel(self.swin2)
        self.ADFF = nn.DataParallel(self.ADFF)
        
        
                
        
        self.optimizer = optim.Adam([
                    {'params': self.swin1.parameters(), 'lr': args.lr, 'betas': (args.beta1, args.beta2), 'weight_decay': args.weight_decay},
                    {'params': self.swin2.parameters(), 'lr': args.lr, 'betas': (args.beta1, args.beta2), 'weight_decay': args.weight_decay},
                    {'params': self.ADFF.parameters(), 'lr': args.lr, 'betas': (args.beta1, args.beta2), 'weight_decay': args.weight_decay},
                ])
        
            
        
        
        self.StepLR = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=30, eta_min=1e-6)
        self.L1Loss = nn.L1Loss(reduction='sum')
        self.mseloss = nn.MSELoss(reduction='sum')
        
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('gpu mode:', self.gpu_mode)
        print('device:', self.device)
        print(torch.cuda.device_count(), 'GPUS!')
        
        if self.gpu_mode:
            self.swin1.to(self.device)
            self.swin2.to(self.device)
            self.ADFF.to(self.device)
            self.L1Loss.to(self.device)
            self.mseloss.to(self.device)
            
    
    
    def train(self):
        self.train_hist = {}
        self.train_hist['train_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        
        if self.load_dump:
            self.load(self.load_path)
            print('continue training!!!!')
        else:
            self.end_epoch = self.epoch
            
        
        print('Training start!!!!')
        start_time = time.time()

        
        for epoch in range(self.start_epoch, self.end_epoch + 1 ):
            
            
            self.swin1.train()
            self.swin2.train()
            self.ADFF.train()
            
            
            print('Epoch: {}'.format(epoch))

            epoch_start_time = time.time()
            
            loss_avg = 0.0
            
            
            max_iter = self.train_data_loader.dataset.__len__() // self.batch_size
            for iter, x in enumerate(tqdm(self.train_data_loader, ncols=80)):
                if self.gpu_mode:
                    inputs = x[0].to(self.device)
                    total_calories = x[2].to(self.device).float()
                    total_mass = x[3].to(self.device).float()
                    total_fat = x[4].to(self.device).float()
                    total_carb = x[5].to(self.device).float()
                    total_protein = x[6].to(self.device).float()
                    inputs_rgbd = x[7].to(self.device)
                    
                 

                self.optimizer.zero_grad()
                
                x_rgb = self.swin1(inputs)
                x_rgbd = self.swin2(inputs_rgbd)
                out = self.C_net(x_rgb, x_rgbd)
                
                
                
                    
                calories_mae = self.L1Loss(out[0], total_calories)
                mass_mae = self.L1Loss(out[1], total_mass)
                fat_mae = self.L1Loss(out[2], total_fat)
                carb_mae = self.L1Loss(out[3], total_carb)
                protein_mae = self.L1Loss(out[4], total_protein)
                
                
                
                total_calories_loss = calories_mae / total_calories.sum().item()
                total_mass_loss = mass_mae / total_mass.sum().item()
                total_fat_loss = fat_mae / total_fat.sum().item()
                total_carb_loss = carb_mae / total_carb.sum().item()
                total_protein_loss = protein_mae / total_protein.sum().item()
                
                
                loss = total_calories_loss + total_mass_loss + total_fat_loss + total_carb_loss + total_protein_loss       
                loss_avg += loss.item()
                
                self.train_hist['train_loss'].append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                
                
            
            print('Epoch: [{:2d}] [{:4d}]/[{:4d}]  loss: {:.8f} '.format(
                    epoch, (iter + 1), max_iter, loss_avg / max_iter
                ))
            
            self.StepLR.step()
            
            
            self.tt(epoch)
           
          
            
        print('Training finish!... save training results')
        self.save(epoch)
        
        self.train_hist['total_time'].append(time.time() - start_time)
        print('AVG one epoch time: {:.2f}, total {} epochs time: {:.2f}'.format(
            np.mean(self.train_hist['per_epoch_time']), self.epoch, self.train_hist['total_time'][0]
        ))
    
    def tt(self, epoch):
        
        self.swin1.eval()
        self.swin2.eval()
        self.ADFF.eval()
            
        calories_loss = 0
        mass_loss = 0
        fat_loss = 0
        carb_loss = 0
        protein_loss = 0
        calories_real = 0
        mass_real = 0
        fat_real = 0
        carb_real =0
        protein_real = 0
        nums = 0
        
        with torch.no_grad():
            for batch_id, x in enumerate(tqdm(self.test_data_loader, ncols=80)):
                if self.gpu_mode:
                    inputs = x[0].to(self.device)
                    total_calories = x[2].to(self.device).float()
                    total_mass = x[3].to(self.device).float()
                    total_fat = x[4].to(self.device).float()
                    total_carb = x[5].to(self.device).float()
                    total_protein = x[6].to(self.device).float()
                    
                    inputs_rgbd = x[7].to(self.device)
                    
                    x_rgb = self.swin1(inputs)
                    x_rgbd = self.swin2(inputs_rgbd)
                    out = self.ADFF(x_rgb, x_rgbd)
               
                
                calories_mae = self.L1Loss(out[0], total_calories)
                mass_mae = self.L1Loss(out[1], total_mass)
                fat_mae = self.L1Loss(out[2], total_fat)
                carb_mae = self.L1Loss(out[3], total_carb)
                protein_mae = self.L1Loss(out[4], total_protein)
                # 求总的绝对误差
                calories_loss += calories_mae.item() 
                mass_loss += mass_mae.item() 
                fat_loss += fat_mae.item() 
                carb_loss += carb_mae.item() 
                protein_loss += protein_mae.item() 
                # 求各类总的真实值
                calories_real += total_calories.sum().item()
                mass_real += total_mass.sum().item()
                fat_real += total_fat.sum().item()
                carb_real += total_carb.sum().item()
                protein_real += total_protein.sum().item()
        
        calories_pmae = calories_loss / calories_real
        mass_pmae = mass_loss / mass_real
        fat_pmae = fat_loss / fat_real
        carb_pmae = carb_loss / carb_real
        protein_pmae = protein_loss / protein_real
        
        mean = (calories_pmae + mass_pmae + fat_pmae + carb_pmae + protein_pmae) / 5
        print('mean:', mean)
        
        if mean < self.min_avg:
            self.save(epoch)
            self.min_avg = mean
    
        
        
    
    def test(self):
        self.load_test(self.args.load)
        
        self.swin1.eval()
        self.swin2.eval()
        self.ADFF.eval()
            
        load_path = self.load_path
        test_r = os.path.join(self.test_path, load_path.stem)
        test_r_p = Path(test_r)
        if not test_r_p.exists():
            test_r_p.mkdir()
            
        calories_loss = 0
        mass_loss = 0
        fat_loss = 0
        carb_loss = 0
        protein_loss = 0
        calories_real = 0
        mass_real = 0
        fat_real = 0
        carb_real =0
        protein_real = 0
        nums = 0
        batch = 0
        csv_rows = []
        with torch.no_grad():
            for batch_id, x in enumerate(tqdm(self.test_data_loader, ncols=80)):
                if self.gpu_mode:
                    inputs = x[0].to(self.device)
                    total_calories = x[2].to(self.device).float()
                    total_mass = x[3].to(self.device).float()
                    total_fat = x[4].to(self.device).float()
                    total_carb = x[5].to(self.device).float()
                    total_protein = x[6].to(self.device).float()
                    inputs_rgbd = x[7].to(self.device)
                    
                    
        
                
                    x_rgb = self.swin1(inputs)
                    x_rgbd = self.swin2(inputs_rgbd)
                    out = self.C_net(x_rgb, x_rgbd)
                    
                           
                    
                nums += out[0].size(0)
                
                calories_mae = self.L1Loss(out[0], total_calories)
                mass_mae = self.L1Loss(out[1], total_mass)
                fat_mae = self.L1Loss(out[2], total_fat)
                carb_mae = self.L1Loss(out[3], total_carb)
                protein_mae = self.L1Loss(out[4], total_protein)
                # 求总的绝对误差
                calories_loss += calories_mae.item() 
                mass_loss += mass_mae.item() 
                fat_loss += fat_mae.item() 
                carb_loss += carb_mae.item() 
                protein_loss += protein_mae.item() 
                # 求各类总的真实值
                calories_real += total_calories.sum().item()
                mass_real += total_mass.sum().item()
                fat_real += total_fat.sum().item()
                carb_real += total_carb.sum().item()
                protein_real += total_protein.sum().item()
               
             
        calories_pmae = calories_loss / calories_real
        mass_pmae = mass_loss / mass_real
        fat_pmae = fat_loss / fat_real
        carb_pmae = carb_loss / carb_real
        protein_pmae = protein_loss / protein_real
        
        calories_mae = calories_loss / nums
        mass_mae = mass_loss / nums
        fat_mae = fat_loss / nums
        carb_mae = carb_loss / nums
        protein_mae = protein_loss / nums

        print('calories_PMAE:', calories_pmae)
        print('mass_PMAE:', mass_pmae)
        print('fat_PMAE:', fat_pmae)
        print('carb_PMAE:', carb_pmae)
        print('protein_PMAE:', protein_pmae)
        print('nums:',nums)
        print('calories_MAE:', calories_mae)
        print('mass_MAE:', mass_mae)
        print('fat_MAE:', fat_mae)
        print('carb_MAE:', carb_mae)
        print('protein_MAE:', protein_mae)
        
        
    
    def save(self, save_epoch):
        save_dir = self.dump_path
        torch.save({
            'swin1': self.swin1.state_dict(),
            'swin2': self.swin2.state_dict(),
            'ADFF': self.ADFF.state_dict(),
            'finish_epoch': save_epoch,
            'result_path': str(save_dir)
        }, (str(save_dir) + '/swin_{}_epoch.pkl'.format(save_epoch)))
        
        
        print("========== save success ===========")
        print('epoch from {} to {}'.format(self.start_epoch, save_epoch))
        print('save result path is {}'.format(str(self.dump_path)))
    
    def load_test(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path))
        self.swin1.load_state_dict(checkpoint['swin1'])
        self.swin2.load_state_dict(checkpoint['swin2'])
        self.ADFF.load_state_dict(checkpoint['ADFF'])
       
    
    def load(self, checkpoint_path):
        checkpoint = torch.load(str(checkpoint_path))
        self.swin1.load_state_dict(checkpoint['swin1'])
        self.ADFF.load_state_dict(checkpoint['ADFF'])
        self.swin2.load_state_dict(checkpoint['swin2'])
        
        self.start_epoch = checkpoint['finish_epoch'] + 1

        self.end_epoch = self.args.epoch +self.start_epoch -1

        print("============== load success =============")
        print("epoch start from {} to {}".format(self.start_epoch, self.end_epoch))
        print('previous result path is {}'.format(checkpoint['result_path']))
    
    def set_seed(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
       
        
       
