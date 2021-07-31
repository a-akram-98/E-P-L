## imports 
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
from dataloader.e2e_dataset import e2e_dataset
from aanet_interface import aanet_interface as aanet_interface
import sys
from torch.utils.tensorboard import SummaryWriter


#from cia_interface import ODModel
def parse_args():
    msg = 'msg'
    parser = argparse.ArgumentParser(msg)
    #parser.add_argument("--lr_od", default=0.0003)
    parser.add_argument("--num_epochs", default=60)
    
    parser.add_argument("--start_epoch", default=0)
    
    parser.add_argument("--batch_size", default=4)
    parser.add_argument("--num_workers", default=8)
    parser.add_argument("--seed", default=42)
    #parser.add_argument("--od_cp", default = "/notebooks/latest.pth", type = str, help = "Loading Weights for Object Detection model")
    #parser.add_argument("--od_config", default = "/notebooks/cia/examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py", type= str
              #         , help = "Configuartion File for the OD model")
    parser.add_argument('--data_dir', default='/notebooks/dataset', type=str, help='Training dataset')
    
    #parser.add_argument('--pkl_path', default='/notebooks/cia/kitti_infos_train.pkl', type=str, help='infos pkl path')
    #parser.add_argument('--load_only_checkpoint' , type = bool, default = True)
    
    parser.add_argument('--Resume_Training' , type = bool, default = False)
    parser.add_argument('--ExperimentName', default='stereoDenseDepth_only', type=str, help='nameOfTheTrainExp')
    
    parser.add_argument('--dp_cp', default='-1.pth', type=str, help='')
    
    parser.add_argument('--aanet_pretrained_path', default='/notebooks/E2E/pretrained', type=str, help='aanet pretrained path')
    
    
    
    parser.add_argument('--COR', default='/notebooks/cor', type=str, help='COR Folder Path')
    args = parser.parse_args()

    return args



def train(args):
    writer = SummaryWriter('/notebooks/logs/depth_only')
    
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device('cuda')
    # DataLoader
    train_data = e2e_dataset(args.data_dir,'train',384,1248)
    train_loader =  DataLoader(dataset=train_data, batch_size=int(args.batch_size), shuffle=True,
                              num_workers=int(args.num_workers), pin_memory=True, drop_last=True)
    AANET_interface = aanet_interface(args.aanet_pretrained_path , args.Resume_Training , args.ExperimentName , args.dp_cp)
    import time
    loss = []
    for epoch in range(args.start_epoch,args.num_epochs):
        
        for i , sample in enumerate(train_loader):
            #t1 = time.time()
            left = sample['left'].to(device)  # [B, 3, H, W]
            right = sample['right'].to(device)
            gt_disp = sample['disp'].to(device)  # [B, H, W]
            
            disp_loss , pred_disp = AANET_interface.aanet_train_batch(left , right , gt_disp)
            loss.append(disp_loss)
            disp_loss.backward()
            AANET_interface.opt_step()
            if i% 50 == 0:
                print("epoch: " , epoch , " iteration: " , i  , " Loss = " , disp_loss, "\n")
            #t2  =  time.time()  - t1 
            #print("iteration took : ", t2)
        
        
        for j,current_loss in enumerate(loss):
                    writer.add_scalar('depth_only loss iterations',
                            current_loss,
                            epoch * len(train_loader) + j)
        writer.add_scalar('depth_only loss epochs',
                            np.sum(loss) / len(train_loader),
                            epoch )
        loss=[]
        AANET_interface.saveCheckpoint(epoch , 0)

if __name__ == "__main__":
    args = parse_args()
    train(args)

