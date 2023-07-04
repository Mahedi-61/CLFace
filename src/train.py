import sys 
import os.path as osp
import os 
import random
import argparse
import numpy as np
import pprint
import torch
import copy

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import  merge_args_yaml
from utils.train_dataset import FacescrubClsDataset
from utils.utils import save_base_model, save_model   
from models import focal_loss
from models.metrics import ArcMarginProduct
from utils.modules import * 
from models.models import AdaFace
import torch.nn.functional as F 


def parse_args():
    """
    Training settings
    For Joint Training (JT) choose option full in 'step' variable
    For Multitask (MT) choose same part to train and test
    """
    parser = argparse.ArgumentParser(description='Compatible with exp: MT, FE, JT')

    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/facescrub.yml',
                        help='optional config file')
    parser.add_argument('--train', type=bool, default=True, help='if train model')
    args = parser.parse_args()
    return args


def get_loss(args):
    if args.model_type == "arcface":
        criterion = focal_loss.FocalLoss(gamma=2)

    elif args.model_type == "adaface":
        criterion = torch.nn.CrossEntropyLoss()

    return criterion


def get_margin(args):
    #cuda + parallel
    if args.model_type == "arcface":
        metric_fc = ArcMarginProduct(args.final_dim, 
                                     args.num_classes, 
                                     s=30, 
                                     m=0.5, 
                                     easy_margin=args.easy_margin)

    elif args.model_type == "adaface":
        metric_fc = AdaFace(embedding_size = args.final_dim, 
                            classnum = args.num_classes) 

    metric_fc.to(args.device)
    metric_fc = torch.nn.DataParallel(metric_fc, device_ids=args.gpu_id)
    return metric_fc


def get_optimizer(args, metric_fc):
    params = [{"params": metric_fc.parameters()}]

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, 
                            lr = args.lr_image_train, 
                            momentum= args.momentum, 
                            weight_decay= args.weight_decay)

    elif args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(params, lr = args.lr_image_train, 
                                      betas=(0.9, 0.999), weight_decay=0.01)

    return optimizer


def get_all_valid_loaders():
    ls_valid_dl = []
    for step in range(0, args.total_step):
        valid_ds = FacescrubClsDataset(transform=None, split="valid", step=step, args=args)
        args.valid_size = valid_ds.__len__()
        valid_dl = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=args.batch_size, 
            drop_last=False,
            num_workers=args.num_workers, 
            shuffle=False)
        
        ls_valid_dl.append(valid_dl)
    return ls_valid_dl



def train_epoch(train_dl, 
                model, 
                metric_fc, 
                metric_fc_old, 
                criterion, 
                optimizer, 
                step, 
                args):
    """
    Training scheme for a epoch
    """
    device = args.device
    metric_fc.train()
    if step > 0: metric_fc_old.eval() 
    total_loss = 0
    ce_loss = 0
    kd_loss = 0

    for imgs, label in train_dl:

        # load cuda
        imgs = imgs.to(device).requires_grad_()
        label = label.to(device)
        
        if args.model_type == "adaface":
            global_feats, _, norm = get_features_adaface(model, imgs)
            y_pred_new = metric_fc(global_feats, norm, label)
            if step > 0: y_pred_old = metric_fc_old(global_feats, norm, label)  

        elif args.model_type == "arcface":
            global_feats = get_features_arcface(model, imgs)
            y_pred_new = metric_fc(global_feats, label) 
            if step > 0: y_pred_old = metric_fc_old(global_feats, label) 

        
        last_class = (step + 1) * args.class_range
        loss_CE = criterion(y_pred_new[:, :last_class], label)

        if step == 0: 
            # At first Inc. step only CE loss
            loss = loss_CE
            ce_loss += loss_CE.item()
            total_loss += loss_CE.item()
        else:
            loss_KD = F.binary_cross_entropy_with_logits(y_pred_new[:, :last_class-args.class_range], 
                                    y_pred_old[:, :last_class-args.class_range].detach().sigmoid()) 
            
            loss = loss_CE + args.KD * loss_KD

            # calculation for display
            total_loss += loss_CE.item() + args.KD * loss_KD.item()
            ce_loss += loss_CE.item()
            kd_loss += args.KD * loss_KD.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
    print(f'\nTraining Epoch [{args.current_epoch}/{args.max_epoch}]', end="")
    print(" | total loss {:0.7f}".format(total_loss / args.train_size), end="")
    if step == 0: print(" | CE loss {:0.7f}".format(ce_loss / args.train_size)) 
    if step > 0:
        print(" | CE loss {:0.7f}".format(ce_loss / args.train_size), end=" ") 
        print(" | KD loss {:0.7f}".format(kd_loss / args.train_size))



def valid_epoch(ls_valid_dl, model, metric_fc, criterion, num_steps, args):
    device = args.device
    model.eval()
    metric_fc.eval()

    with torch.no_grad():
        for task in range(0, num_steps + 1):

            total_loss = 0
            correct = 0
            for imgs, label in ls_valid_dl[task]:

                # load cuda
                imgs = imgs.to(device).requires_grad_()
                label = label.to(device)
                
                if args.model_type == "adaface":
                    global_feats, _, norm = get_features_adaface(model, imgs)
                    output = metric_fc(global_feats, norm, label) 

                elif args.model_type == "arcface":
                    global_feats = get_features_arcface(model, imgs)
                    output = metric_fc(global_feats, label) 
                
                #first_class = task * args.class_range
                last_class = (task + 1) * args.class_range        
                loss = criterion(output[:, :last_class], label)
                total_loss += loss.item()

                out_ind = torch.argmax(output, axis=1)
                correct += sum(out_ind == label)

            acc = correct / args.valid_size
            val_loss = total_loss / args.valid_size
            print("Task {:2d}: val acc {:0.8f}| loss {:0.8f}".format(task, acc, val_loss))



def main(args):
    LR_change_seq = [2, 4, 6]

    #load model (cuda + parallel + grd. false + eval)
    if args.model_type == "adaface":   model = prepare_adaface(args) 
    elif args.model_type == "arcface": model = prepare_arcface(args)

    metric_fc = get_margin(args)
    
    opt = get_optimizer(args, metric_fc)    
    criterion = get_loss(args)
    ls_valid_dl = get_all_valid_loaders()

    #pprint.pprint(args)
    print("\nstart training ...")
    
    for step in range(0, args.total_step):
        print(f"\n\n############### Incremental Step: {step} ####################")

        train_ds = FacescrubClsDataset(transform=None, split="train", step=step, args=args)
        args.train_size = train_ds.__len__()

        train_dl = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=args.batch_size, 
            drop_last=False,
            num_workers=args.num_workers, 
            shuffle=True)

        del train_ds 
        lr = args.lr_image_train
        
        # Frozen old model for calculating knowledge distribution (KD) loss
        # At first step no KD loss
        if step == 0: metric_fc_old = None
        else:
            metric_fc_old = copy.deepcopy(metric_fc)
            for p in metric_fc_old.parameters():
                p.requires_grad = False
    
        for epoch in range(1, args.max_epoch + 1):
            args.current_epoch = epoch
        
            train_epoch(train_dl, model, metric_fc, metric_fc_old, criterion, opt, step, args)

            if epoch in LR_change_seq: 
                for g in opt.param_groups:
                    lr = lr * args.gamma
                    g['lr'] = lr 
                    print("Learning Rate change to: {:0.5f}".format(lr))
        
            if step + 1 == args.total_step:
                if epoch % args.save_interval==0:
                    save_model(metric_fc, epoch, args)

            if ((args.do_valid == True) and (epoch % args.valid_interval == 0) and (epoch != 0)):
                valid_epoch(ls_valid_dl, model, metric_fc, criterion,  step, args)
        

if __name__ == "__main__":
    args = merge_args_yaml(parse_args())

    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    args.device = torch.device("cuda")
    
    main(args)