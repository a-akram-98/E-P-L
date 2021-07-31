import sys
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
sys.path.append('/notebooks/Notebooks/deepPruner/DeepPruner/deeppruner')
from models.deeppruner import DeepPruner
from loss_evaluation import loss_evaluation
class dp_interface():
    def __init__(self ,pretrained = None,save = None, mode='train'): #, isFast=True):
        self.save = save
        self.model = DeepPruner()
        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        if pretrained!=None:
            state_dict = torch.load(pretrained)
            self.model.load_state_dict(state_dict['state_dict'], strict=True)
            print("Loading " + pretrained + "\n")
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00002, betas=(0.9, 0.999))
        if mode =='train':
            self.model.train()
        else:
            self.model.eval()
            
            
    def dp_test_batch(self, left,right ,gt_size):
        with torch.no_grad():
            imgL = Variable(torch.FloatTensor(left))
            imgR = Variable(torch.FloatTensor(right))
        #disp_L = Variable(torch.FloatTensor(disp_L))
            imgL, imgR = imgL.cuda(), imgR.cuda()
            pred_disp = self.model(imgL, imgR)
        return pred_disp
    def dp_train_batch(self,left , right , gt_disp):
        imgL = Variable(torch.FloatTensor(left))
        imgR = Variable(torch.FloatTensor(right))
        disp_L = Variable(torch.FloatTensor(gt_disp))
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
        mask = (disp_true > 0)
        mask.detach_()
        self.optimizer.zero_grad()
        result = self.model(imgL, imgR)

        loss, _ = loss_evaluation(result, disp_true, mask,8 )

        #loss.backward()
        #optimizer.step()

        return loss , result[0]
    def opt_step(self):
        self.optimizer.step()
        
        
    def saveCheckpoint(self , epoch , iteration):
        torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict()}, os.path.join(self.save , "E2E_{}.tar".format(epoch)))