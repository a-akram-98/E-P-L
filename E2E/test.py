import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np
from dataloader.e2e_dataset_val import e2e_dataset
from aanet_interface import aanet_interface as aanet_interface
import sys

from det3d.torchie.trainer.trainer import example_to_device
############### for map ########################
#from get_results_interface import evaluate
from det3d.datasets.utils.kitti_object_eval_python.evaluate import (evaluate as kitti_evaluate,)
#####################for getting label detections #############################
from det3d.datasets.kitti.kitti import  KittiDataset
##########################################
from det3d.datasets.kitti import kitti_common as kitti_common
from  det3d.torchie.parallel.collate import collate_kitti
from cia_interface import ODModel
def parse_args():
    msg = 'msg'
    parser = argparse.ArgumentParser(msg)
    parser.add_argument("--num_workers", default=8)
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--od_cp", default="/notebooks/E2E/pretrained/cia/epoch_80.pth", type=str,
                        help="Loading Weights for Object Detection model")
    parser.add_argument("--od_config",
                        default="/notebooks/cia/examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py",
                        type=str
                        , help="Configuartion File for the OD model")
    parser.add_argument('--data_dir', default='/notebooks/dataset', type=str, help='Training dataset')
    parser.add_argument('--pkl_path', default='/notebooks/cia/kitti_infos_val.pkl', type=str, help='infos pkl path')
    parser.add_argument('--load_only_checkpoint', type=bool, default=True)
    parser.add_argument('--Resume_Training', type=bool, default=False)
    parser.add_argument('--ExperimentName', default='depth_only_1', type=str, help='nameOfTheTestExp')
    parser.add_argument('--dp_cp', default='15_0.pth', type=str, help='')
    parser.add_argument('--aanet_pretrained_path', default='/notebooks/E2E/pretrained', type=str, help='aanet pretrained path')
    parser.add_argument('--COR', default='/notebooks/cor', type=str, help='COR Folder Path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    sys.path.append(args.COR)
    from cor_interface import COR
    device = torch.device('cuda')

    ###################################################################### needs alki's modifications ####################
    test_dataset = e2e_dataset(args.data_dir, 'val', 384, 1248)
    test_loader = DataLoader(dataset=test_dataset, batch_size=int(args.batch_size), shuffle=True,
                             num_workers=int(args.num_workers), pin_memory=True, drop_last=True)
    
    #########################################################################################################################

    ###################################################################### no car solution #################################
    #### Temp solve the cia not car infos
    import os
    workDir = "/notebooks/"
    LabelDir = '/notebooks/dataset/training/label_2/'
    filenames = open("/notebooks/dataset/val.txt").read().split('\n')

    def read_label(label_filename):
        lines = [line.rstrip() for line in open(label_filename)]
        return lines

    import pickle
    def getinfos(infos_path):
        with open(infos_path, 'rb') as f:
            data = pickle.load(f)
        return data

    infos = getinfos("/notebooks/cia/kitti_infos_val.pkl")
    count = 0
    non_car_filenames_idx = []

    for i, file in enumerate(filenames):
        Objects = read_label(os.path.join(LabelDir, file + ".txt"))
        has_car = False
        # print(Objects)
        for j, obj in enumerate(Objects):
            if obj.split()[0] == "Car":
                has_car = True
                break
        # print(has_car)
        if has_car == True:
            # count = count+1
            non_car_filenames_idx.append(i)
            # txt_file.write(file + "\n")
    infos = [infos[i] for i in non_car_filenames_idx]
    #with open('val_no_ped.pkl', 'wb') as handle:
    #    pickle.dump(infos, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #exit()
    ########################################################################################################################

    ############################################## instantion of models #####################################################
    AANET_interface = aanet_interface(args.aanet_pretrained_path, args.Resume_Training, args.ExperimentName, args.dp_cp ,mode='val')
    CIA_interface = ODModel(cfg_path=args.od_config, loader_len=len(test_loader), cp_pth=args.od_cp,
                            load_only_checkpoint=args.load_only_checkpoint)
    CIA_interface.register_aanet(AANET_interface)
    pipeline = CIA_interface.get_pipline()
    ########################################################################################################################

    ########################################################################################################################
    # there must be a way here to load the checkpoint of the e2e total model here
    ########################################################################################################################

    #######################################################testhng the model ###############################################
    CIA_interface.model.eval()
    cpu_device = torch.device("cpu")
    detections = {}
    
    kitti = KittiDataset("notebooks/dataset/","/notebooks/cia/val_no_ped.pkl" , class_names = ['car'])#,pipeline=pipeline )
    import time
    for i, sample in enumerate(test_loader):  # note that we enumerate on samples not batches here

        with torch.no_grad():
            left = sample['left'].to(device)  # [B, 3, H, W]
            right = sample['right'].to(device)
            #gt_disp = sample['disp'].to(device)  # [B, H, W]
            #t0 = time.time()
            ############################################################################################################
            pred_disp = AANET_interface.aanet_test_batch(left, right,(384, 1248))  ##need to be changed to test batch
            #t1 = time.time()
            ############################################################################################################
            #t2 = time.time()
            ############################################################################################################
            batch = COR(infos, sample["idx"], pred_disp, pipeline)
            #exit()
            #t3 = time.time()
            # cor output is already torch and on cuda so it is passed directly to cia (needs to be checked)
            #batch[0]["num_points"] = torch.from_numpy(batch[0]["num_points"])
            #batch[0]["num_voxels"] = torch.from_numpy(batch[0]["num_voxels"])
            #batch[0]["coordinates"] = torch.from_numpy(batch[0]["coordinates"])
            #batch[0]["calib"] = torch.from_numpy(batch[0]["calib"])
            batch = collate_kitti(batch)
            
            example = example_to_device(batch, device=device)  # This was used to pass numpy point cloud to torch !
            # output = CIA_interface.cia_forward(batch, epoch, i)   this can't be used !
            #print(batch)
            if i%99 == 0:
                print("current: " ,i,"/" , len(test_loader)  , "\n")
            
            outputs = CIA_interface.model(example, return_loss=False, rescale=True)  # list_length=batch_size: 8
            
            for output in outputs:  # output.keys(): ['box3d_lidar', 'scores', 'label_preds', 'metadata']
                token = output["metadata"]["token"]  # token should be the image_id
                for k, v in output.items():
                    if k not in ["metadata", ]:
                        output[k] = v.to(cpu_device)
                detections.update({token: output, })
            #t4 = time.time()
            #print(t1-t0 ," ", t2-t1, " " , t3-t2 , " " , t4-t3)
            #print(t3-t2)
            #exit()

    _, detections = kitti.evaluation(detections,partial=False,output_dir= None)
    a_file = open("data.pkl", "wb")
    pickle.dump(detections, a_file)
    a_file.close()
    res_dir = os.path.join(workDir, "predictions")
    os.makedirs(res_dir, exist_ok=True)
    
    for dt in detections:
        with open(os.path.join(res_dir, "%06d.txt" % int(dt["metadata"]["token"])), "w") as fout:
            lines = kitti_common.annos_to_kitti_label(dt)
            for line in lines:
                fout.write(line + "\n")

    LabelDir = '/notebooks/dataset/training/label_2/'
    label_split_file = "/notebooks/dataset/val_no_ped.txt"
    ap_result_str, ap_dict = kitti_evaluate(LabelDir, res_dir, label_split_file=label_split_file, current_class=0,)
    print(ap_result_str)


if __name__ == "__main__":
    main()
