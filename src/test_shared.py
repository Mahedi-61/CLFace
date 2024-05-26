import sys 
import os.path as osp
import random
import argparse
import numpy as np
import pprint
import torch
from tqdm import tqdm 

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import merge_args_yaml
from utils.test_dataset import FacescrubTestDataset   
from utils.modules import * 
from models.models import AdaFace


def parse_args():
    parser = argparse.ArgumentParser(description='for backbone (frozen) + shared arch.')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/lfw.yml',
                        help='optional config file')
    args = parser.parse_args()
    return args


def test(test_dl, 
         backbone, 
         shared_net, 
         task, 
         args):
    
    device = args.device
    shared_net.eval()
    backbone.eval()

    labels = []
    preds = [] 
    with torch.no_grad():
        for img1, img2, label in test_dl:

            # load cuda
            img1 = img1.to(device).requires_grad_()
            img2 = img2.to(device).requires_grad_()
            label = label.to(device)
            
            if args.model_type == "adaface":
                global_feats, _, norm = get_features_adaface(model, img1)

            elif args.model_type == "arcface":
                global_feats_1, _ = get_shared_features_arcface(backbone, shared_net, img1)
                global_feats_2, _ = get_shared_features_arcface(backbone, shared_net, img2)

                cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
                pred = cosine_sim(global_feats_1, global_feats_2)
                preds += pred.data.cpu().tolist()
                labels += label.data.cpu().tolist()

    calculate_scores(preds, labels, args)
    calculate_identification_acc(preds, labels, args)



def get_all_test_loaders(args):
    ls_test_dl = []

    for step in range(0, args.total_step):
        test_ds = FacescrubTestDataset(transform=None, task=step, args=args)
        args.test_size = test_ds.__len__()
        test_dl = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=args.batch_size, 
            drop_last=False,
            num_workers=args.num_workers, 
            shuffle=False)
        
        ls_test_dl.append(test_dl)
    return ls_test_dl




def main(args):

    #load model (cuda + parallel + grd. false + eval)
    if args.model_type == "adaface":   backbone, shared_net = prepare_adaface_shared(args)
    elif args.model_type == "arcface": backbone, shared_net = prepare_arcface_shared(args) 

    test_loader = get_all_test_loaders(args)
    #pprint.pprint(args)
    print("\nstart testing ...")
    
    if args.is_unseen:
        begin_step = args.seen_steps 
        end_step = args.total_step
    else:
        begin_step = 0
        end_step = args.seen_steps

    for step in range(begin_step, end_step):
        print(f"\n\n############### Incremental Task: {step} ####################")

        test_dl = test_loader[step]
        test(test_dl, backbone, shared_net, step, args)
        


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