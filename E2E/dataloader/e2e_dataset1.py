from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import AAnet_transforms
from torch.utils.data import Dataset
from utils import utils
from utils.file_io import read_img, read_disp

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class e2e_dataset(Dataset):
    def __init__(self ,data_dir,mode):
        self.mode = mode
        train_transform_list = [AAnet_transforms.RandomCrop(args.img_height, args.img_width),
                            AAnet_transforms.RandomColor(),
                            AAnet_transforms.RandomVerticalFlip(),
                            AAnet_transforms.ToTensor(),
                            AAnet_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]
        self.train_transform = AAnet_transforms.Compose(train_transform_list)
        val_transform_list = [transforms.RandomCrop(args.val_img_height, args.val_img_width, validate=True),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                         ]
        val_transform = transforms.Compose(val_transform_list)
        self.data_dir = data_dir
        
        kitti_2015_dict = {
            'train': 'filenames/KITTI_2015_train.txt',
            'train_all': 'filenames/KITTI_2015_train_all.txt',
            'val': 'filenames/KITTI_2015_val.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }
        data_filenames = kitti_2015_dict[mode]
        lines = utils.read_text_lines(data_filenames)
        self.samples = []
        for line in lines:
            splits = line.split()
            left_img, right_img = splits[:2]
            gt_disp =  splits[2]
            
            sample['left_name'] = left_img.split('/', 1)[1]
            
            sample['left'] = os.path.join(data_dir, left_img)
            sample['right'] = os.path.join(data_dir, right_img)
            sample['disp'] = os.path.join(data_dir, gt_disp)
            sample['pseudo_disp'] = None
        self.samples.append(sample)
        
    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]
        sample['left_name'] = sample_path['left_name']

        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])


        if sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'], subset=False)  # [H, W]
        if self.mode == 'train':
            sample = self.train_transform(sample)
        else:
            sample = self.val_transform(sample)
        return sample
    
    def __len__(self):
        return len(self.samples)