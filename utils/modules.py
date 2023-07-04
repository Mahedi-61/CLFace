import sys
import os.path as osp
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms
import os, random  

from utils.utils import load_base_model, load_shared_model
from models import net
from models.ir_resnet import iresnet18
from utils.dataset_utils import *

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        return x if random.random() > self.p else self.fn(x)


def get_dataset_specific_transform(args, train=True):
    if args.dataset_name == "cifar100" and train == True:
        trans = transforms.Compose([
        transforms.Pad(2),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        #RandomApply(transforms.GaussianBlur((3, 3), (1.5, 1.5)), p=0.1),
        transforms.RandomCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]))
        ])

    elif args.dataset_name == "cifar100" and train == False:
        trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]))
        ])
    
    else:
        trans = None 
    return trans 


############   features   ############
def get_features_arcface(model, imgs):
    img_features = model(imgs)
    return img_features

def get_shared_features_arcface(backbone, shared_net, imgs):
    x = backbone(imgs)
    img_features = shared_net(x)
    feat = shared_net.module[0](x)
    feat.requires_grad_(True) 
    return img_features, feat 

def get_features_adaface(model, imgs):
    _, img_features, norm = model(imgs)
    return img_features, _, norm 


def get_tpr(fprs, tprs):
    fpr_val = [10 ** -4, 10 ** -3]
    tpr_fpr_row = []
    for fpr_iter in np.arange(len(fpr_val)):
        _, min_index = min(
            list(zip(abs(fprs - fpr_val[fpr_iter]), range(len(fprs)))))
        tpr_fpr_row.append(tprs[min_index] * 100)
    return tpr_fpr_row


################## scores ####################
def calculate_scores(y_score, y_true, args):
    # sklearn always takes (y_true, y_pred)
    fprs, tprs, threshold = metrics.roc_curve(y_true, y_score)
 
    fprs = np.flipud(fprs)
    tprs = np.flipud(tprs)

    eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]
    auc = metrics.auc(fprs, tprs)
    tpr_fpr_row = get_tpr(fprs, tprs)
    print("AUC {:.4f} | EER {:.4f} | TPR@FPR=1e-4 {:.4f} | TPR@FPR=1e-3 {:.4f}".format(
        auc, eer, tpr_fpr_row[0], tpr_fpr_row[1]))

    if args.is_roc == True:
        data_dir = "."
        with open(os.path.join(data_dir, args.roc_file + '.npy'), 'wb') as f:
            np.save(f, y_true)
            np.save(f, y_score)



def calculate_identification_acc(preds, labels, args):
    max_range = args.class_range * args.class_range
    total_acc = 0

    for i in range(0, args.num_imposter_per_sub):
        y_score = preds[i*max_range : (i+1)*max_range]
        y_true = labels[i*max_range : (i+1)*max_range]

        total_number = int(np.sqrt(len(y_score)))

        y_score = np.array(y_score).reshape((total_number, total_number))
        y_score = np.argmax(y_score, axis=1)

        y_true = np.array(y_true).reshape((total_number, total_number))
        y_true = np.argmax(y_true, axis=1)

        #print(y_score)
        total_acc += sum([1 for i, j in zip(y_score, y_true) if i==j]) / total_number
    
    print("identification accuracy (%)", (total_acc/args.num_imposter_per_sub) * 100)


###########   model   ############
"""
# no pretrain for CL learning. (set it always False in cfg file)
if args.is_pretrain == True:
    model_path = args.arcface_pretrain_path
    weights = torch.load(model_path)
    model.load_state_dict(weights)
    print("Loading pretrianed ArcFace model")
"""
def prepare_arcface_base(args):
    device = args.device
    model = iresnet18(pretrained=False, progress=True)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    return model


def prepare_arcface(args):
    device = args.device
    model = iresnet18(pretrained=False, progress=True)

    # base weights for CL learning. (set it False in cfg file while training base)
    if args.is_base == True:
        model_path = args.arcface_base_path
        model = load_base_model(model, model_path)
        print("Loading base weights")

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    for p in model.parameters():
        p.requires_grad = False
    return model 


def prepare_arcface_shared(args):
    device = args.device
    model = iresnet18(pretrained=False, progress=True)

    model_path = args.arcface_base_path
    model = load_base_model(model, model_path)
    print("Loading base weights")

    base_net = nn.Sequential(*list(model.children())[:6])

    start_layer = 9
    if args.dataset_name == "cifar100": start_layer = 10
    shared_net = nn.Sequential(*[*list(model.children())[6:9], 
                 nn.Flatten(), *list(model.children())[start_layer:]])

    base_net.to(device)
    base_net = torch.nn.DataParallel(base_net, device_ids=args.gpu_id)
    for p in base_net.parameters():
        p.requires_grad = False


    if args.CONFIG_NAME == "test":
        model_path = args.arcface_test_path
        shared_net = load_shared_model(shared_net, model_path)
        for p in shared_net.parameters():
            p.requires_grad = False
        print("loading .. shared network for test: ", args.arcface_test_path) 


    elif args.CONFIG_NAME == "shared":
        for p in shared_net.parameters():
            p.requires_grad = True
    
    shared_net.to(device)
    shared_net = torch.nn.DataParallel(shared_net, device_ids=args.gpu_id)
    return base_net, shared_net 



#######lossess ##########333
def kl_div(n_img, out, prev_out, T=2):
    """
    Compute the knowledge-distillation (KD) loss with soft targets.

    Parameters
        ----------
        target_outputs : ,required
            Outputs from the frozen Net A
        outputs : , required
            Outputs of the original classes from the adaptive Net B
        n_img: int, required
            Number of images from the new m classes in a mini-batch
        T : optional
            Temperature factor normally set to 2
    """
    log_p = torch.log_softmax(out / T, dim=1)
    q = torch.softmax(prev_out / T, dim=1)
    res = torch.nn.functional.kl_div(log_p, q, reduction='none')
    res = res.sum() / n_img
    return res


def grad_cam_loss(feature_o, out_o, feature_n, out_n):
    batch = out_n.size()[0]
    index = out_n.argmax(dim=-1).view(-1, 1)
    onehot = torch.zeros_like(out_n)
    onehot.scatter_(-1, index, 1.)
    out_o, out_n = torch.sum(onehot * out_o), torch.sum(onehot * out_n)
    
    grads_o = torch.autograd.grad(out_o, feature_o, allow_unused=True)[0]
    print(grads_o)
    print(grads_o.shape)
    grads_n = torch.autograd.grad(out_n, feature_n, create_graph=True, allow_unused=True)[0]
    weight_o = grads_o.mean(dim=(2, 3)).view(batch, -1, 1, 1)
    weight_n = grads_n.mean(dim=(2, 3)).view(batch, -1, 1, 1)
    
    cam_o = F.relu((feature_o * weight_o).sum(dim=1)) #grads_o
    cam_n = F.relu((feature_n * weight_n).sum(dim=1)) #grads_n
    
    # normalization
    cam_o = F.normalize(cam_o.view(batch, -1), p=2, dim=-1)
    cam_n = F.normalize(cam_n.view(batch, -1), p=2, dim=-1)
    
    loss_AD = (cam_o - cam_n).norm(p=1, dim=1).mean()
    return loss_AD


#### model for Ada Face 
def prepare_adaface(args):
    device = args.device
    architecture = "ir_18"

    args.load_model_path = args.adaface_path 
    model = net.build_model(architecture)
    
    statedict = torch.load(args.load_model_path)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print("Loading pretrianed AdaFace model")
    return model 



if __name__ == "__main__":
    from easydict import EasyDict as edict
    args = edict()
    model_path = "."
    model = iresnet18(pretrained=False, progress=True)
    backbone = nn.Sequential(*list(model.children())[:6])
    