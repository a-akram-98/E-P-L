from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import sys
import dataloader.AAnet_transforms as AAnet_transforms
from torch.utils.data import Dataset
from utils.utils import read_img, read_disp , get_depth
from utils.kitti_util import get_depth_map , read_label
import pickle
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class e2e_dataset(Dataset):
    def __init__(self ,dataset_path,mode , img_height , img_width , applyAllTransforms=False):
        self.mode = mode
        self.samples_paths = []
        sample = dict()
        self.img_height = img_height
        self.img_width = img_width
        
        leftImagesDir = dataset_path + '/training/image_2/'
        rigthImagesDir = dataset_path +  '/training/image_3/'
        VelodyneDir = dataset_path + '/training/velodyne/'
        CalibDir = dataset_path + '/training/calib/'
        LabelDir = dataset_path + '/training/label_2/'
        DepthDir =dataset_path+ '/training/disparities/'
        if mode == 'train':
            if applyAllTransforms == False:
                transform_list = [AAnet_transforms.RandomCrop(img_height,  img_width)
                              ,AAnet_transforms.ToTensor(),
                              AAnet_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
            else:
                transform_list = [AAnet_transforms.RandomCrop(img_height, img_width),
                            AAnet_transforms.RandomColor(),
                            AAnet_transforms.RandomVerticalFlip(),
                            AAnet_transforms.ToTensor(),
                            AAnet_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]
            
            filenames = open(dataset_path + "/train_no_ped.txt").read().split('\n')

        else:
            filenames = open(dataset_path + "/val_no_ped.txt").read().split('\n')
            transform_list = [AAnet_transforms.RandomCrop(img_height, img_width,validate=True),                               AAnet_transforms.ToTensor(),
                              AAnet_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        self.transform = AAnet_transforms.Compose(transform_list)
    
        for name in filenames:
                sample = dict()
                sample['left_name'] = name
                sample['left']  = leftImagesDir  + name + '.png'
                sample['right']  = rigthImagesDir  + name + '.png'
                sample['velo']  =  VelodyneDir + name + '.bin'
                sample['calib'] = CalibDir + name + '.txt'
                sample['label'] = LabelDir + name + '.txt'
                sample['disp'] = DepthDir + name + '.npy'
                self.samples_paths.append(sample)
        
    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples_paths[index]
        sample['left_name'] = sample_path['left_name']
        sample['idx'] = index
        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])

        #sample['disp'] =get_depth(sample_path['disp']) #get_depth_map(index , self.samples_paths)
        
        
        #label

        #sample['label'] = torch.Tensor(read_label(index , self.samples_paths))
        
        sample = self.transform(sample)

        
        return sample
    def getinfos(self, infos_path):
        with open(infos_path, 'rb') as f:
            data = pickle.load(f)
        return data
    def __len__(self):
        return len(self.samples_paths)
