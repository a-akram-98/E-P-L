## imports 
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
from dataloader.e2e_dataset import e2e_dataset
from aanet_interface import aanet_interface as aanet_interface
import sys

from cia_interface import ODModel
def parse_args():
    msg = 'msg'
    parser = argparse.ArgumentParser(msg)
    parser.add_argument("--lr_od", default=0.0003)
    parser.add_argument("--num_epochs", default=10)
    parser.add_argument("--batch_size", default=2)
    parser.add_argument("--num_workers", default=8)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--od_cp", default = "/notebooks/latest.pth", type = str, help = "Loading Weights for Object Detection model")
    parser.add_argument("--od_config", default = "/notebooks/cia/examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py", type= str
                       , help = "Configuartion File for the OD model")
    parser.add_argument('--data_dir', default='/notebooks/dataset', type=str, help='Training dataset')
    
    parser.add_argument('--pkl_path', default='/notebooks/cia/kitti_infos_train.pkl', type=str, help='infos pkl path')
    parser.add_argument('--load_only_checkpoint' , type = bool, default = True)
    
    parser.add_argument('--Resume_Training' , type = bool, default = False)
    parser.add_argument('--ExperimentName', default='denseDepthTrain', type=str, help='nameOfTheTrainExp')
    
    parser.add_argument('--dp_cp', default='2_0.pth', type=str, help='')
    
    parser.add_argument('--aanet_pretrained_path', default='/notebooks/E2E/pretrained', type=str, help='aanet pretrained path')
    
    
    
    parser.add_argument('--COR', default='/notebooks/cor', type=str, help='COR Folder Path')
    args = parser.parse_args()

    return args


def train(args):
    sys.path.append(args.COR)
    from cor_interface import COR  
    # For reproducibility
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device('cuda')
    # DataLoader
    train_data = e2e_dataset(args.data_dir,'train',384,1248)
    train_loader =  DataLoader(dataset=train_data, batch_size=int(args.batch_size), shuffle=True,
                              num_workers=int(args.num_workers), pin_memory=True, drop_last=True)

    
    #print(args.batch_size)
    
    #### Temp solve the cia not car infos

    
    infos = getModifiedInfos(args)
    print(args.Resume_Training)
    AANET_interface = aanet_interface(args.aanet_pretrained_path , args.Resume_Training , args.ExperimentName , args.dp_cp)
    CIA_interface =  ODModel( cfg_path = args.od_config, loader_len = len(train_loader), cp_pth = args.od_cp, load_only_checkpoint = args.load_only_checkpoint)
    CIA_interface.register_aanet(AANET_interface)
    pipeline = CIA_interface.get_pipline()
    
#     torch.autograd.set_detect_anomaly(True)
    CIA_interface.model.train()
#     for param in AANET_interface.aanet.parameters():
#         param.requires_grad = False
    for epoch in range(CIA_interface.epoch() , args.num_epochs):
        CIA_interface.before_train_epoch()
        for i , sample in enumerate(train_loader):
            
            left = sample['left'].to(device)  # [B, 3, H, W]
            right = sample['right'].to(device)
            gt_disp = sample['disp'].to(device)  # [B, H, W]

            #################### The pipeline of pseudo lidar ####################
            disp_loss , pred_disp = AANET_interface.aanet_train_batch(left , right , gt_disp)
            
            
            batch = COR(infos, sample["idx"], pred_disp ,pipeline )
            
            output =  CIA_interface.cia_forward(batch ,epoch ,i)
            #######################################################################
            
            total_loss = output["loss"] + 0.07*disp_loss
            CIA_interface.after_train_iter(disp_loss, total_loss)
            total_loss.backward()
            CIA_interface.clip_grad()
            CIA_interface.optimizer_step()
            AANET_interface.opt_step()
        CIA_interface.after_train_epoch()

        
        
        
def getModifiedInfos(args):
    import os
    LabelDir ='/notebooks/dataset/training/label_2/'# args.data_dir + '/training/label_2/'
    filenames = open("/notebooks/dataset/train.txt").read().split('\n')
    def read_label(label_filename):
        lines = [line.rstrip() for line in open(label_filename)]
        return lines
    import pickle
    def getinfos( infos_path):
        with open(infos_path, 'rb') as f:
            data = pickle.load(f)
        return data
    infos = getinfos("/notebooks/cia/kitti_infos_train.pkl")
    count = 0
    non_car_filenames_idx = []

    for i , file in enumerate(filenames):
        Objects = read_label(os.path.join(LabelDir, file + ".txt"))
        has_car = False
        #print(Objects)
        for j , obj in enumerate(Objects):
            if obj.split()[0] == "Car":
                has_car = True
                break
        #print(has_car)
        if has_car == True:
            #count = count+1
            non_car_filenames_idx.append(i)
            #txt_file.write(file + "\n")
    infos = [infos[i] for i in non_car_filenames_idx]
    return infos    
        
        

if __name__ == "__main__":
    args = parse_args()
    train(args)