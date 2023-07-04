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
    * This file supports base training of all datasets
    * Change the cfg file for specific dataset
    For Multitask (MT) choose same part to train and test
    """
    parser = argparse.ArgumentParser(description='Compatible with all datasets for base training')

    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/cifar100.yml',
                        help='optional config file')
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
                                     s=args.s, 
                                     m=args.m, 
                                     easy_margin=args.easy_margin)

    elif args.model_type == "adaface":
        metric_fc = AdaFace(embedding_size = args.final_dim, 
                            classnum = args.num_classes) 

    metric_fc.to(args.device)
    metric_fc = torch.nn.DataParallel(metric_fc, device_ids=args.gpu_id)
    return metric_fc


def get_optimizer(args, model, metric_fc):
    #params = {"params": itertools.chain(*[model.parameters(), metric_fc.parameters()])}
    params = [{'params': model.parameters()}, {'params': metric_fc.parameters()}]

    if args.optimizer == 'sgd':
        print("loading sgd")
        
        optimizer = torch.optim.SGD(params, 
                            lr = args.lr_image_train, 
                            momentum= args.momentum, 
                            weight_decay= args.weight_decay)
        
    elif args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(params, lr = args.lr_image_train, betas=(0.9, 0.999), weight_decay=0.01)

    return optimizer


def get_all_valid_loaders():
    trans = get_dataset_specific_transform(args, train=False)
    valid_ds = FacescrubClsDataset(transform=trans, split="base_valid", step=0, args=args)
    args.valid_size = valid_ds.__len__()
   
    valid_dl = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=args.batch_size, 
        drop_last=False,
        num_workers=args.num_workers, 
        shuffle=False)

    return valid_dl



def train_epoch(train_dl, 
                model, 
                metric_fc, 
                criterion, 
                optimizer, 
                args):
    """
    Training scheme for a epoch
    """
    device = args.device
    metric_fc.train()
    model.train()
    total_loss = 0

    for imgs, label in train_dl:

        # load cuda
        imgs = imgs.to(device).requires_grad_()
        label = label.to(device)
        
        if args.model_type == "adaface":
            global_feats, _, norm = get_features_adaface(model, imgs)
            y_pred_new = metric_fc(global_feats, norm, label)

        elif args.model_type == "arcface":
            global_feats = get_features_arcface(model, imgs)
            y_pred_new = metric_fc(global_feats, label) 

        
        last_class = args.class_range
        loss_CE = criterion(y_pred_new[:, :last_class], label)

        loss = loss_CE
        total_loss += loss_CE.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'\nTraining Epoch [{args.current_epoch}/{args.max_epoch}]', end="")
    print(" | total loss {:0.7f}".format(total_loss / args.train_size))


def valid_epoch(ls_valid_dl, model, metric_fc, criterion,  args):
    device = args.device
    model.eval()
    metric_fc.eval()

    with torch.no_grad():
        total_loss = 0
        correct = 0
        for imgs, label in ls_valid_dl:

            # load cuda
            imgs = imgs.to(device).requires_grad_()
            label = label.to(device)
            
            if args.model_type == "adaface":
                global_feats, _, norm = get_features_adaface(model, imgs)
                output = metric_fc(global_feats, norm, label) 

            elif args.model_type == "arcface":
                global_feats = get_features_arcface(model, imgs)
                output = metric_fc(global_feats, label) 
            
            last_class = args.class_range        
            loss = criterion(output[:, :last_class], label)
            total_loss += loss.item()

            out_ind = torch.argmax(output, axis=1)
            correct += sum(out_ind == label)

        acc = correct / args.valid_size
        val_loss = total_loss / args.valid_size
        print("Base training: val acc {:0.8f}| loss {:0.8f}".format(acc, val_loss))


def main(args):
    # config for two mode (base and IL for full_fozen)
    LR_change_seq = args.lr_base_training

    #load model (cuda + parallel + grd. false + eval)
    if args.model_type == "adaface":   model = prepare_adaface_base(args) 
    elif args.model_type == "arcface": model = prepare_arcface_base(args)

    metric_fc = get_margin(args)
    
    opt = get_optimizer(args, model, metric_fc)    
    criterion = get_loss(args)
    ls_valid_dl = get_all_valid_loaders()

    #pprint.pprint(args)
    print("\nstart training ...")
        
    trans = get_dataset_specific_transform(args, train=True)
    train_ds = FacescrubClsDataset(transform=trans, split="base", step=0, args=args)
    args.train_size = train_ds.__len__()

    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        drop_last=False,
        num_workers=args.num_workers, 
        shuffle=True)

    del train_ds 
    lr = args.lr_image_train
    
    for epoch in range(1, args.max_epoch + 1):
        args.current_epoch = epoch
    
        train_epoch(train_dl, model, metric_fc, criterion, opt, args)

        if epoch % args.save_interval==0:
            save_base_model(model, epoch, args)

        if ((args.do_valid == True) and (epoch % args.valid_interval == 0) and (epoch != 0)):
            valid_epoch(ls_valid_dl, model, metric_fc, criterion, args)

        if epoch in LR_change_seq:
            lr = lr * args.gamma 
            for g in opt.param_groups:
                g['lr'] = lr 
            print("Learning Rate change to: {:0.5f}".format(lr))
        

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