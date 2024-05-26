import sys
import os.path as osp
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import os, random  
from tqdm import tqdm 

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

from utils.utils import load_base_model, load_shared_model, load_full_model
from models import net
from models.ir_resnet import iresnet18
from models.metrics import ArcMarginProduct
from models.models import AdaFace
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        return x if random.random() > self.p else self.fn(x)



def pod_loss(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="spatial",
    normalize=True,
    **kwargs):
    """Pooled Output Distillation.
    Reference:
        * Douillard et al.
          Small Task Incremental Learning.
          arXiv 2020.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :return: A float scalar loss.
    """
    assert len(list_attentions_a) == len(list_attentions_b)

    loss = torch.tensor(0.).to(my_device)
    #for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
    a = list_attentions_a
    b = list_attentions_b

    assert a.shape == b.shape, (a.shape, b.shape)

    a = torch.pow(a, 2)
    b = torch.pow(b, 2)

    if collapse_channels == "channels":
        a = a.sum(dim=1).view(a.shape[0], -1)  # shape of (b, w * h)
        b = b.sum(dim=1).view(b.shape[0], -1)

    elif collapse_channels == "width":
        a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * h)
        b = b.sum(dim=2).view(b.shape[0], -1)

    elif collapse_channels == "height":
        a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * w)
        b = b.sum(dim=3).view(b.shape[0], -1)

    elif collapse_channels == "gap":
        a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0]
        b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]

    elif collapse_channels == "spatial":
        a_h = a.sum(dim=3).view(a.shape[0], -1)
        b_h = b.sum(dim=3).view(b.shape[0], -1)
        a_w = a.sum(dim=2).view(a.shape[0], -1)
        b_w = b.sum(dim=2).view(b.shape[0], -1)
        a = torch.cat([a_h, a_w], dim=-1)
        b = torch.cat([b_h, b_w], dim=-1)
    else:
        raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

    if normalize:
        a = F.normalize(a, dim=1, p=2)
        b = F.normalize(b, dim=1, p=2)

    layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
    loss = layer_loss #+

    return loss / len(list_attentions_a)


def get_gpkd(global_feats, global_feats_old):
    global_feats = F.normalize(global_feats, p=2, dim=0)
    global_feats_old = F.normalize(global_feats_old, p=2, dim=0)

    return nn.CosineEmbeddingLoss()(global_feats, 
                                    global_feats_old.detach(), 
                                    torch.ones(global_feats.shape[0]).to(my_device)) 

    

############   features   ############
def get_shared_features_arcface(backbone, shared_net, imgs):
    x = backbone(imgs)
    img_features = shared_net(x)
    feat = shared_net.module[0](x)
    feat.requires_grad_(True) 
    return img_features, feat 


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
            
            img1 = img1.to(my_device)
            img2 = img2.to(my_device)

            img1_h = img1_h.to(my_device)
            img2_h = img2_h.to(my_device)
            pair_label = pair_label.to(my_device)

            # get global and local image features from COTS model
            if args.model_type == "arcface":
                global_feat1, local_3 = model(img1)
                global_feat2, local_3 = model(img2)

                global_feat1_h, local_3  = model(img1_h)
                global_feat2_h, local_3  = model(img2_h)

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
    model = iresnet18(pretrained=False, progress=True)
    model.to(my_device)

    if args.is_base == True:
        print("Backbone network (fully trainable + No pretrain weights)")

    elif args.is_base == False:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model'])
        print("Loading saved backbone network weights from: ", args.model_path)

    return model


def prepare_margin(args, ):
    if args.model_type == "arcface":
        metric_fc = ArcMarginProduct(args.final_dim, 
                                        args.base_num, 
                                        s= args.s, 
                                        margin= args.m)

    elif args.model_type == "adaface":
        metric_fc = AdaFace(embedding_size = args.final_dim, 
                            classnum = args.base_num) 

    metric_fc.to(my_device)

    if args.is_base == True:
        print("ArcFace network (fully trainable + No pretrain weights)")

    elif args.is_base == False:
        checkpoint = torch.load(args.model_path)
        metric_fc.load_state_dict(checkpoint['metric_fc'])
        print("Loading saved metric_fc weights from: ", args.model_path)

    return metric_fc



def prepare_adaface(args):
    architecture = "ir_18"
    model = net.build_model(architecture)
    model.to(my_device)
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

    model.to(my_device)
    return model 




def prepare_arcface_shared(args):
    model = iresnet18(pretrained=False, progress=True)

    model_path = args.arcface_base_path
    model = load_base_model(model, model_path)
    print("Loading base weights")

    base_net = nn.Sequential(*list(model.children())[:6])

    start_layer = 9
    if args.dataset_name == "cifar100": start_layer = 10
    shared_net = nn.Sequential(*[*list(model.children())[6:9], 
                 nn.Flatten(), *list(model.children())[start_layer:]])

    base_net.to(my_device)
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
    
    shared_net.to(my_device)
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
def prepare_adaface_vjal(args):
    architecture = "ir_18"

    args.load_model_path = args.adaface_path 
    model = net.build_model(architecture)
    
    statedict = torch.load(args.load_model_path)['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    
    model.to(my_device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print("Loading pretrianed AdaFace model")
    return model 


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



if __name__ == "__main__":
    from easydict import EasyDict as edict
    args = edict()
    model_path = "."
    model = iresnet18(pretrained=False, progress=True)
    backbone = nn.Sequential(*list(model.children())[:6])
    