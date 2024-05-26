import os, sys
import os.path as osp
import torch
import numpy as np
import torch
from tqdm import tqdm 
import copy

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.dataset import TrainDataset, TestDataset
from models import focal_loss
from utils.modules import * 

my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CLFace:
    def __init__(self, args):
        self.args = args
        num = args.num_classes // 2 
        self.base_num = num - (num % args.step_size)
        self.class_range = self.base_num // args.step_size 
        
        self.args.base_num = self.base_num
        self.args.class_range = self.class_range
        self.val_acc = 0

        # by design
        if self.args.is_base: 
            self.args.step = 0
            self.args.freeze = 0

        self.get_dataloaders()

        if self.args.model_type == "arcface":
            self.criterion = focal_loss.FocalLoss(gamma=2)
            self.model = prepare_arcface(self.args)

        elif self.args.model_type == "adaface":
            self.criterion = torch.nn.CrossEntropyLoss()
            self.model = prepare_adaface(self.args) 

        self.metric_fc = prepare_margin(self.args)
        self.get_optimizer()    


    def save_model(self, ):
        save_dir = os.path.join(self.args.checkpoint_path, 
                                self.args.dataset, 
                                self.args.setup,
                                str(self.args.step_size),
                                "step%d" % self.args.step)
        
        os.makedirs(save_dir, exist_ok=True)

        name = '%s_18_ms1m_step%d.pth' % (self.args.model_type, 
                                           self.args.step) 
        
        state_path = os.path.join(save_dir, name)
        state = {'model': self.model.state_dict(), 
                 "metric_fc": self.metric_fc.state_dict()}
        
        torch.save(state, state_path)
        print("saving ... ", name)


    def get_dataloaders(self, ):
        if self.args.train == True:
            self.args.split = "train"
            train_ds = TrainDataset(self.args) 
            self.args.train_size = train_ds.__len__()

            self.train_dl = torch.utils.data.DataLoader(
                train_ds, 
                batch_size=self.args.batch_size, 
                drop_last=False,
                num_workers=self.args.num_workers, 
                shuffle=True)

            self.args.ver_list = os.path.join("./data", self.args.test_dataset, 
                                              "images", self.args.test_file)
            self.args.split = "valid"
            del train_ds

        elif self.args.train == False:
            self.args.ver_list = os.path.join("./data", self.args.test_dataset, 
                                              "images", self.args.test_file)
            self.args.split = "test"

        valid_ds = TestDataset(self.args)
        self.valid_dl = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=self.args.batch_size, 
            drop_last=False,
            num_workers=self.args.num_workers, 
            shuffle=False)

        del valid_ds


    def get_optimizer(self, ):
        ss = 25 if self.args.is_base == True else 3
        lr_new = self.args.lr_train if self.args.is_base == True else 0.01

        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                        lr = lr_new, 
                                        momentum= self.args.momentum, 
                                        weight_decay= self.args.weight_decay)

        self.optimizer_fc = torch.optim.SGD(self.metric_fc.parameters(), 
                                        lr = lr_new, 
                                        momentum= self.args.momentum, 
                                        weight_decay= self.args.weight_decay)
        
        self.lrs_optimizer = torch.optim.lr_scheduler.StepLR(
                                        self.optimizer, 
                                        step_size = ss,
                                        gamma=0.9997)
        
        self.lrs_optimizer_fc = torch.optim.lr_scheduler.StepLR(
                                        self.optimizer_fc, 
                                        step_size = ss,
                                        gamma=0.9997)


    def train_epoch(self, ):
        print("\n")
        self.metric_fc.train()
        self.model.train()
        
        print_loss = {}
        print_loss["ce_loss"] = 0
        print_loss["gpkd_loss"] = 0
        print_loss["pod_loss"] = 0
        loop = tqdm(total = len(self.train_dl))

        for i, (imgs, label) in enumerate(self.train_dl):
            imgs = imgs.to(my_device) 
            label = label.to(my_device)
            
            if self.args.model_type == "adaface":
                global_feats, _, norm = self.model(imgs)
                y_pred_new = self.metric_fc(global_feats, norm, label)

            elif self.args.model_type == "arcface":
                global_feats, local_3 = self.model(imgs)
                global_feats_old, local_3_old = self.model_old(imgs)
                y_pred_new = self.metric_fc(global_feats, label) 
            
            last_class = self.args.class_range

            if self.args.is_base == True:
                loss_CE = self.criterion(y_pred_new, label)

            elif self.args.is_base == False:
                last_class = self.args.step * self.args.class_range
                loss_CE = self.criterion(y_pred_new[:, :last_class], label)
                loss_gpkd = self.args.delta_gpkd * get_gpkd(global_feats, global_feats_old)
                loss_pod = 0.0  #self.args.delta_pod * pod_loss(local_3_old, local_3)
                
                loss = loss_CE + loss_gpkd + loss_pod
                print_loss["ce_loss"] += loss_CE.item()
                print_loss["gpkd_loss"] +=  0.0 #loss_gpkd.item() 
                print_loss["pod_loss"] +=  0.0 #loss_pod.item()

            self.optimizer.zero_grad()
            self.optimizer_fc.zero_grad()
            loss.backward()

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
        print("Focal Loss {:0.7f}".format(print_loss["ce_loss"] / self.args.train_size))
        print("GPKD  Loss {:0.7f}".format(print_loss["gpkd_loss"]  / self.args.train_size))
        print("POD  Loss {:0.7f}".format(print_loss["pod_loss"]  / self.args.train_size))

        print("lr model: ", self.lrs_optimizer.get_lr())
        print("lr metric_fc: ", self.lrs_optimizer_fc.get_lr())


    """
    if step == 0: 
        # At first Inc. step only CE loss
        loss = loss_CE
        ce_loss += loss_CE.item()
        total_loss += loss_CE.item()
    else:
        loss_KD = F.binary_cross_entropy_with_logits(y_pred_new[:, :last_class-args.class_range], 
                                y_pred_old[:, :last_class-args.class_range].detach().sigmoid()) 
    """
    
    def train(self,):
        print("\nstart training ...")
        self.model_old = copy.deepcopy(self.model)
        for p in self.model_old.parameters():
            p.requires_grad = False
        self.model_old.eval()

        for epoch in range(self.args.max_epoch):
            self.args.current_epoch = epoch
            self.train_epoch()
            
            if epoch > 4:
                acc = valid_epoch(self.valid_dl, self.model, self.args)

                if acc > self.val_acc:
                    self.val_acc = acc
                    self.save_model()

            if epoch == 11 and self.args.is_base == True:
                self.lrs_optimizer.gamma = 0.9994
                self.lrs_optimizer_fc.gamma = 0.9994
                print("chaning gamma value of lr scheduler")

        print("Best val_acc: {:.5f}".format(self.val_acc))


    def test(self, ):
        print("\n **** start testing ****")
        valid_epoch(self.valid_dl, self.model, self.args)