from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader.dataset import Nutrition_RGBD
import pdb
import random


def get_DataLoader(args):
    train_transform = transforms.Compose([
        transforms.Resize((args.size+50, args.size+100)),
        transforms.RandomCrop((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.size+50, args.size+100)),
        transforms.CenterCrop((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    nutrition_rgbd_ims_root = os.path.join(args.data_root, 'imagery')
    nutrition_train_txt = os.path.join(args.data_root, 'imagery', 'rgbd_train_processed.txt')
    nutrition_test_txt = os.path.join(args.data_root, 'imagery', 'rgbd_test_processed1.txt')
    nutrition_train_rgbd_txt = os.path.join(args.data_root, 'imagery', 'rgb_in_overhead_train_processed.txt')
    nutrition_test_rgbd_txt = os.path.join(args.data_root, 'imagery', 'rgb_in_overhead_test_processed1.txt')
       
    trainset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_train_rgbd_txt, nutrition_train_txt,  transform=train_transform)
    testset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_test_rgbd_txt, nutrition_test_txt, transform=test_transform)
        
    
    train_loader = DataLoader(trainset,
                              batch_size=args.b,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True
                              )
    test_loader = DataLoader(testset,
                             batch_size=args.b,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True
                             )

    return train_loader, test_loader
