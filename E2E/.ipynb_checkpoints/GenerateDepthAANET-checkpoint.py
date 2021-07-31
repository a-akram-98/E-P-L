import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
from dataloader.e2e_dataset import e2e_dataset
from aanet_interface import aanet_interface as aanet_interface
import sys

#from det3d.torchie.trainer.trainer import example_to_device
############### for map ########################
#from get_results_interface import evaluate
#from det3d.datasets.utils.kitti_object_eval_python.evaluate import (evaluate as kitti_evaluate,)
#####################for getting label detections #############################
#from det3d.datasets.kitti.kitti import  KittiDataset
##########################################
#from det3d.datasets.kitti import kitti_common as kitti_common
#from  det3d.torchie.parallel.collate import collate_kitti
#from cia_interface import ODModel
def parse_args():
    msg = 'msg'
    parser = argparse.ArgumentParser(msg)
    parser.add_argument("--num_workers", default=8)
    parser.add_argument("--batch_size", default=1)
   
    
    parser.add_argument('--data_dir', default='/notebooks/dataset', type=str, help='Training dataset')
    
    parser.add_argument('--load_only_checkpoint', type=bool, default=True)
    parser.add_argument('--Resume_Training', type=bool, default=False)
    parser.add_argument('--ExperimentName', default='', type=str, help='nameOfTheTestExp')
    parser.add_argument('--dp_cp', default='15_0.pth', type=str, help='')
    
    parser.add_argument('--aanet_pretrained_path', default='/notebooks/E2E/pretrained', type=str, help='aanet pretrained path')
   

    
    parser.add_argument('--save_path', default='/notebooks/GeneratedDepthForImediateTrain_1/', type=str, help='aanet pretrained path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda')
    
    test_dataset = e2e_dataset(args.data_dir, 'train', 384, 1248)
    test_loader = DataLoader(dataset=test_dataset, batch_size=int(args.batch_size), shuffle=True,
                             num_workers=int(args.num_workers), pin_memory=True, drop_last=True)
    

    AANET_interface = aanet_interface(args.aanet_pretrained_path, args.Resume_Training, args.ExperimentName, args.dp_cp ,mode='train')

    for i, sample in enumerate(test_loader):  # note that we enumerate on samples not batches here
        with torch.no_grad():
            left = sample['left'].to(device)  # [B, 3, H, W]
            right = sample['right'].to(device)
            pred_disp = AANET_interface.aanet_test_batch(left, right,(384, 1248))  ##need to be changed to test batch
            np.save( args.save_path +sample['left_name'][0] + ".npy" , pred_disp.squeeze(0).cpu().numpy() )
            if i%99 == 0:
                print("current: " ,i,"/" , len(test_loader)  , "\n")
            
            
            
            



if __name__ == "__main__":
    main()
