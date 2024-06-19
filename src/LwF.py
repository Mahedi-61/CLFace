import os, sys
import os.path as osp
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm 
from torch.optim import Adam

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.modules import * 
from base_train import BaseTrainer


class LwM (BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

    def train_epoch(self, ):
        print("\n")
        beta = 1
        self.metric_fc.train()
        self.model.train()
        self.model_old.train()
        total_loss = 0
        loop = tqdm(total = len(self.train_dl))

        for i, (imgs, label) in enumerate(self.train_dl):
            imgs = imgs.to(self.my_device) 
            label = label.to(self.my_device)

            if self.args.model_type == "arcface":
                global_feats, l_2, l_3, l_4 = self.model(imgs)
                y_pred_new = self.metric_fc(global_feats, label) 

                global_feats_old, l_2, l_3, l_4_old = self.model_old(imgs)
                y_pred_old = self.metric_fc_old(global_feats_old, label) 
                l_4_old.requires_grad_(True)

            if self.args.is_base == False:
                last_class = self.args.step * self.args.class_range
                y_pred_new = y_pred_new[:, :last_class]
                y_pred_old = y_pred_old[:, :last_class - self.args.class_range]

                loss_CE = self.criterion(y_pred_new, label)

                if self.args.step > 1:
                    loss_D = F.binary_cross_entropy_with_logits(y_pred_new[:, :-self.class_range], 
                                                                y_pred_old.detach().sigmoid())
                    
                elif self.args.step == 1: 
                    loss_D = 0

                loss = loss_CE + loss_D * beta

            self.optimizer.zero_grad()
            self.optimizer_fc.zero_grad()
            loss.backward()
            total_loss += loss_CE.item()

            self.optimizer_fc.step()
            if (self.args.current_epoch >= self.args.freeze):
                self.optimizer.step()
                self.lrs_optimizer.step()
                self.lrs_optimizer_fc.step()

            # update loop information
            loop.update(1)
            loop.set_description(f'Training Epoch [{self.args.current_epoch + 1}/{self.args.max_epoch}]')
            loop.set_postfix()

        loop.close()
        print("total loss {:0.7f}".format(total_loss / self.args.train_size))
        print("lr model: ", self.lrs_optimizer.get_last_lr())
        print("lr metric_fc: ", self.lrs_optimizer_fc.get_last_lr())


    def train(self,):
        print("\nstart training ...")
        self.model_old = copy.deepcopy(self.model)
        self.model_old.requires_grad_(False)
        
        self.metric_fc_old = copy.deepcopy(self.metric_fc)
        self.metric_fc_old.requires_grad_(False)

        for epoch in range(self.args.max_epoch):
            self.args.current_epoch = epoch
            self.train_epoch()
            
            if epoch > 3:
                acc = valid_epoch(self.valid_dl, self.model, self.args)

                if acc > self.val_acc:
                    self.val_acc = acc
                    save_model(self.args, self.model, self.metric_fc)

            if epoch == 10 and self.args.is_base == True:
                self.lrs_optimizer.gamma = 0.9994
                self.lrs_optimizer_fc.gamma = 0.9994
                print("chaning gamma value of lr scheduler")

        return self.val_acc