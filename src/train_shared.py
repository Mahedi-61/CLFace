import sys 
import os.path as osp
import random
import argparse
import numpy as np
import pprint
import torch
import os 
import copy

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import  merge_args_yaml
from utils.train_dataset import FacescrubClsDataset
from utils.utils import save_shared_models  
from models import focal_loss
from models.metrics import ArcMarginProduct
from utils.modules import * 
from models.models import AdaFace
import torch.nn.functional as F 


def parse_args():
    parser = argparse.ArgumentParser(description='for backbone (frozen) + shared arch.')
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



def get_optimizer(args, shared_net, metric_fc):  
    ## if you add more parameter into the group then change opt.param_groups[1]["lr"]
    params = [{"params": shared_net.parameters(), "lr": args.lr_shared_layer}, 
              {"params": metric_fc.parameters()}]

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, 
                                    lr = args.lr_image_train, 
                                    momentum = args.momentum, 
                                    weight_decay = args.weight_decay)


    elif args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(params, 
                                      lr = args.lr_image_train, 
                                      betas = (0.9, 0.999), 
                                      weight_decay = args.weight_decay)

    return optimizer



def get_all_valid_loaders(args):
    ls_valid_dl = []
    ls_valid_size = []
    trans = get_dataset_specific_transform(args, train=False)

    for step in range(0, args.total_step):
        valid_ds = FacescrubClsDataset(transform=trans, split="valid", step=step, args=args)
        ls_valid_size.append(valid_ds.__len__())
        valid_dl = torch.utils.data.DataLoader(
            valid_ds, 
            batch_size=args.batch_size, 
            drop_last=False,
            num_workers=args.num_workers, 
            shuffle=True)
        
        ls_valid_dl.append(valid_dl)

    return ls_valid_dl, ls_valid_size



def train_epoch(train_dl, 
          backbone, 
          shared_net,
          shared_net_old, 
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
    backbone.eval()
    shared_net.train()
    metric_fc.train()

    if step > 0:
        shared_net_old.eval() 
        metric_fc_old.eval()

    total_loss = 0
    ce_loss = 0
    kd_loss = 0
    kl_div_loss = 0

    for imgs, label in train_dl:

        # load cuda
        imgs = imgs.to(device).requires_grad_()
        label = label.to(device)
        
        if args.model_type == "adaface":
            global_feats, _, norm = get_features_adaface(backbone, shared_net, imgs)
            y_pred_new = metric_fc(global_feats, norm, label)

            if step > 0: 
                gl_feats_old, _, norm = get_shared_features_adaface(backbone, shared_net_old, imgs)
                y_pred_old = metric_fc_old(gl_feats_old, norm, label)  

        elif args.model_type == "arcface":
            global_feats, feat = get_shared_features_arcface(backbone, shared_net, imgs)
            y_pred_new = metric_fc(global_feats, label)

            if step > 0: 
                gl_feats_old, feat_old = get_shared_features_arcface(backbone, shared_net_old, imgs)
                y_pred_old = metric_fc_old(gl_feats_old, label)  

        last_class = (step + 1) * args.class_range
        loss_CE = criterion(y_pred_new[:, :last_class], label)

        if step == 0: 
            # At first Inc. step only CE loss
            loss = loss_CE
            total_loss += loss_CE.item()
            ce_loss += loss_CE.item()

        else:
            loss_KD = F.binary_cross_entropy_with_logits(y_pred_new[:, :last_class-args.class_range], 
                                    y_pred_old[:, :last_class-args.class_range].detach().sigmoid()) 
            
            loss_kl_div = kl_div(args.batch_size, global_feats, gl_feats_old, T=args.temperature)

            """
            loss_AD = grad_cam_loss(feat_old, 
                                    y_pred_old[:, :last_class-args.class_range], 
                                    feat, 
                                    y_pred_new[:, :last_class-args.class_range])
            """
            
            loss = loss_CE + (args.KD * loss_KD) + (args.kl_div * loss_kl_div)  #+ args.att_distill * loss_AD 
            ce_loss += loss_CE.item()
            kd_loss += args.KD * loss_KD.item()
            kl_div_loss += args.kl_div * loss_kl_div.item() 


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
    

    print(f'\nTraining Epoch [{args.current_epoch}/{args.max_epoch}]', end="")
    print(" | total loss {:0.7f}".format(total_loss / args.train_size), end="")
    if step == 0: print(" | CE loss {:0.7f}".format(ce_loss  / args.train_size)) 
    if step > 0:
        print(" | CE loss {:0.7f}".format(ce_loss / args.train_size), end=" ") 
        print(" | KD loss {:0.7f}".format(kd_loss / args.train_size), end=" ")
        print(" | KL DIV {:0.7f}".format(kl_div_loss / args.train_size))


def valid_epoch(ls_valid_dl, ls_valid_size, backbone, shared_net, metric_fc, criterion, num_steps, args):
    device = args.device
    backbone.eval()
    shared_net.eval()
    metric_fc.eval()
    avg_increm_acc = []

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
                    global_feats, _ = get_shared_features_arcface(backbone, shared_net, imgs)
                    output = metric_fc(global_feats, label) 
                
                #first_class = task * args.class_range
                last_class = (task + 1) * args.class_range        
                loss = criterion(output[:, :last_class], label)
                total_loss += loss.item()

                out_ind = torch.argmax(output, axis=1)
                correct += sum(out_ind == label)

            acc = correct / ls_valid_size[task]
            avg_increm_acc.append(acc)
            val_loss = total_loss / ls_valid_size[task]
            print("Task {:2d}: val acc {:0.5f}| loss {:0.5f}".format(task, acc, val_loss))

    print("Average Incremental Accuracy: {:0.5f}".format(sum(avg_increm_acc) / len(avg_increm_acc)))


def main(args):
    LR_change_seq = args.lr_shared_training

    #load model (cuda + parallel + grd. false + eval)
    if args.model_type == "adaface":   backbone, shared_net = prepare_adaface_shared(args)
    elif args.model_type == "arcface": backbone, shared_net = prepare_arcface_shared(args) 

    metric_fc = get_margin(args)
    opt = get_optimizer(args, shared_net, metric_fc)  
    criterion = get_loss(args)
    ls_valid_dl, ls_valid_size = get_all_valid_loaders(args)

    #pprint.pprint(args)
    print("\nstart training ...")
    trans = get_dataset_specific_transform(args, train=True)

    for step in range(0, args.total_step):
        print(f"\n\n############### Incremental Step: {step} ####################")

        train_ds = FacescrubClsDataset(transform=trans, split="train", step=step, args=args)
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
        if step == 0: 
            shared_net_old = None 
            metric_fc_old = None
        else:
            shared_net_old = copy.deepcopy(shared_net)
            metric_fc_old = copy.deepcopy(metric_fc)

            for p in metric_fc_old.parameters():
                p.requires_grad = False
            for p in shared_net_old.parameters():
                p.requires_grad = False
    
        for epoch in range(1, args.max_epoch + 1):
            torch.cuda.empty_cache()
            args.current_epoch = epoch
            
            train_epoch(train_dl, backbone, shared_net, shared_net_old, 
                        metric_fc, metric_fc_old, criterion, opt, step, args)

            if step + 1 == args.total_step:
                if epoch % args.save_interval==0:
                    save_shared_models(shared_net, metric_fc, epoch, args)

            if ((args.do_valid == True) and (epoch % args.valid_interval == 0) and (epoch != 0)):
                valid_epoch(ls_valid_dl, ls_valid_size, backbone, shared_net, metric_fc, criterion,  step, args)

            if epoch in LR_change_seq: 
                lr = lr * args.gamma
                opt.param_groups[1]['lr'] = lr 
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