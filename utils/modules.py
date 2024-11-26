import sys
import os.path as osp
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import os, random  
from tqdm import tqdm 
from torch.autograd import Variable 

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from utils.utils import load_base_model, load_full_model
from models import net
from models.ir_resnet import iresnet18, iresnet50
from models.metrics import ArcMarginProduct
from models.models import AdaFace


def save_model(args, model, metric_fc):
    save_dir = os.path.join(args.checkpoint_path, 
                            args.dataset, 
                            args.setup,
                            str(args.step_size),
                            "step%d" % args.step)
    
    os.makedirs(save_dir, exist_ok=True)

    name = '50_%s_%s_%s_step%d.pth' % (args.model_type, args.arch, args.dataset, args.step) 
    
    state_path = os.path.join(save_dir, name)
    state = {'model': model.state_dict(), 
             "metric_fc": metric_fc.state_dict()}
    
    torch.save(state, state_path)
    print("saving ... ", name)



def get_msd_loss(
    list_attentions_a,
    list_attentions_b,
    args):

    normalize=True    
    assert len(list_attentions_a) == len(list_attentions_b)
    loss = torch.tensor(0.).to(args.my_device)

    for a, b in zip(list_attentions_a, list_attentions_b):
        assert a.shape == b.shape, (a.shape, b.shape)

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
        b = b.sum(dim=1).view(b.shape[0], -1)

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss 

    return loss / len(list_attentions_a)


def get_gpkd(global_feats, global_feats_old, args):
    global_feats = F.normalize(global_feats, p=2, dim=0)
    global_feats_old = F.normalize(global_feats_old, p=2, dim=0)

    return nn.CosineEmbeddingLoss()(global_feats, 
                                    global_feats_old.detach(), 
                                    torch.ones(global_feats.shape[0]).to(args.my_device)) 

    

############   features   ############
def get_tpr(fprs, tprs):
    fpr_val = [10 ** -4, 10 ** -3]
    tpr_fpr_row = []
    for fpr_iter in np.arange(len(fpr_val)):
        _, min_index = min(
            list(zip(abs(fprs - fpr_val[fpr_iter]), range(len(fprs)))))
        tpr_fpr_row.append(tprs[min_index] * 100)
    return tpr_fpr_row


def valid_epoch(valid_dl, model, args):
    model.eval()
    preds = []
    labels = []

    loop = tqdm(total = len(valid_dl))
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    with torch.no_grad():
        for  data in valid_dl:
            img1, img2, img1_h, img2_h, pair_label = data 
            
            img1 = img1.to(args.my_device)
            img2 = img2.to(args.my_device)

            img1_h = img1_h.to(args.my_device)
            img2_h = img2_h.to(args.my_device)
            pair_label = pair_label.to(args.my_device)
        
            # get global and local image features from COTS model
            if args.model_type == "arcface":
                global_feat1, l_2, l_3, l_4 = model(img1)
                global_feat2, l_2, l_3, l_4 = model(img2)

                global_feat1_h, l_2, l_3, l_4  = model(img1_h)
                global_feat2_h, l_2, l_3, l_4  = model(img2_h)

            elif args.model_type == "adaface":
                global_feat1,  _, norm = model(img1)
                global_feat2,  _, norm = model(img2)

                global_feat1_h,  _, norm = model(img1_h)
                global_feat2_h,  _, norm = model(img2_h)

            gf1 = torch.cat((global_feat1, global_feat1_h), dim=1)
            gf2 = torch.cat((global_feat2, global_feat2_h), dim=1)

            pred = cosine_sim(gf1, gf2)
            preds += pred.data.cpu().tolist()
            labels += pair_label.data.cpu().tolist()

            # update loop information
            loop.update(1)
            loop.set_postfix()

    loop.close()
    acc, _ = calculate_acc(preds, labels, args)
    return acc 
    

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
def prepare_arcface(args, ):
    if args.arch == "18":
        model = iresnet18(pretrained=False, progress=True)

    elif args.arch == "50":
        model = iresnet50(pretrained=False, progress=True)

    model.to(args.my_device)

    if args.is_base == True:
        print("Backbone network (fully trainable + No pretrain weights)")

    elif args.is_base == False:
        checkpoint = torch.load(args.model_path, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        #checkpoint = torch.load("weights/pretrain/arcface_ir50_ms1mv3.pth", weights_only=True)
        #model.load_state_dict(checkpoint)
        print("Loading saved backbone network weights from: ", args.model_path)

    return model


def prepare_margin(args, ):
    if args.model_type == "arcface":
        metric_fc = ArcMarginProduct(args.final_dim, 
                                        args.base_num, 
                                        args.my_device,
                                        s= args.s, 
                                        margin= args.m)

    elif args.model_type == "adaface":
        metric_fc = AdaFace(embedding_size = args.final_dim, 
                            classnum = args.base_num) 

    metric_fc.to(args.my_device)

    if args.is_base == True:
        print("ArcFace network (fully trainable + No pretrain weights)")

    elif args.is_base == False:
        checkpoint = torch.load(args.model_path, weights_only=True)
        metric_fc.load_state_dict(checkpoint['metric_fc'])
        print("Loading saved metric_fc weights from: ", args.model_path)

    return metric_fc


def prepare_adaface():
    architecture = "ir_18"
    model = net.build_model(architecture)
    #model.to(my_device)
    print("AdaFace network (fully trainable + No pretrain weights)")
    return model


def prepare_adaface_pretrained(args):
    architecture = "ir_18"
    model = net.build_model(architecture)
    
    # base weights for CL learning.
    if args.is_base == True:
        model_path = args.arcface_base_path
        model = load_base_model(model, model_path)
        print("Loading base weights")
    
    elif args.is_base == False:
        model_path = args.arcface_full_model_path
        model = load_full_model(model, model_path)
        print("Loading full model weights from: ", args.arcface_full_model_path)

    #model.to(my_device)
    return model 



#######lossess ##########
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


def KFold(n, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        frac = int(n/n_folds)
        test = base[i * frac : (i + 1) * frac]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[0]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[1]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def calculate_acc(preds, labels, args):
    predicts = []
    num_imgs = len(preds)
    with torch.no_grad():
        for i in range(num_imgs):

            #distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append('{}\t{}\n'.format(preds[i], labels[i]))

    accuracy = []
    thd = []
    folds = KFold(n=num_imgs, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.01)
    predicts = np.array(list(map(lambda line: line.strip('\n').split(), predicts)))

    print(len(predicts))
    print("starting fold ....")
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
        #print("finding fold: ", idx)

    print('ACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), 
                                np.std(accuracy), np.mean(thd)))
    return np.mean(accuracy), predicts


def get_ckd_loss(std_emb, tea_emb, eps=1e-8, temp3=2.0):

    batch_size = std_emb.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size))).to(std_emb.device)

    if std_emb.dim() == 2:
        std_emb = std_emb.unsqueeze(0)
        tea_emb = tea_emb.unsqueeze(0)

    std_emb_norm = torch.norm(std_emb, 2, dim=2, keepdim=True)
    tea_emb_norm = torch.norm(tea_emb, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(std_emb, tea_emb.transpose(1, 2))
    norm0 = torch.bmm(std_emb_norm, tea_emb_norm.transpose(1, 2))
    scores0 = scores0 / (norm0.clamp(min=eps) * temp3)

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    return loss0 


if __name__ == "__main__":
    from easydict import EasyDict as edict
    args = edict()
    model_path = "."
    model = iresnet18(pretrained=False, progress=True)
    backbone = nn.Sequential(*list(model.children())[:6])
    
