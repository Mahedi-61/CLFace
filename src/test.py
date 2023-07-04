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
from utils.utils import merge_args_yaml, load_test_models
from utils.train_dataset import FacescrubClsDataset   
from models.metrics import ArcMarginProduct
from utils.modules import * 
from models.models import AdaFace


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='FE')
    parser.add_argument('--cfg', dest='cfg_file', type=str, 
                        default='./cfg/facescrub.yml',
                        help='optional config file')
    args = parser.parse_args()
    return args

def get_margin(args):
    if args.model_type == "arcface":
        metric_fc = ArcMarginProduct(args.final_dim, 
                                            args.num_classes, 
                                            s=30, m=0.5, 
                                            easy_margin=args.easy_margin)

    elif args.model_type == "adaface":
        metric_fc = AdaFace(embedding_size = args.final_dim, 
                            classnum = args.num_classes) #cuda + parallel

    metric_fc.to(args.device)
    metric_fc = torch.nn.DataParallel(metric_fc, device_ids=args.gpu_id)
    return metric_fc


def test(test_dl, model, metric_fc, args):
    device = args.device
    model.eval()
    metric_fc.eval()
    correct = 0

    with torch.no_grad():
        loop = tqdm(total=len(test_dl))

        for imgs, label in test_dl:

            # load cuda
            imgs = imgs.to(device)
            label = label.to(device)
            
            if args.model_type == "adaface":
                global_feats, _, norm = get_features_adaface(model, imgs)
                output = metric_fc(global_feats, norm, label) 

            elif args.model_type == "arcface":
                global_feats = get_features_arcface(model, imgs)
                output = metric_fc(global_feats, label) 

            out_ind = torch.argmax(output, dim=1)
            correct += sum(out_ind == label)

            # update loop information
            loop.update(1)
            loop.set_postfix()
            del global_feats

    loop.close()
    str_loss = "accuracy {:0.4f}".format(correct / args.test_size)
    print(str_loss)



def main(args):
    # prepare dataloader, models, data
    test_ds = FacescrubClsDataset(transform=None, split="test", args=args)

    args.test_size = test_ds.__len__()
    test_dl = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        drop_last=False,
        num_workers=args.num_workers, 
        shuffle=False)

    del test_ds

    if args.model_type == "adaface":
        model = prepare_adaface(args) #cuda + parallel + grd. false + eval
    elif args.model_type == "arcface":
        model = prepare_arcface(args)

    metric_fc = get_margin(args)

    for epoch in range(10, 20):
        test_model_path = "./checkpoints/facescrub/FE/arcface/part_5/arcface_part_5_epoch_%d.pth" % epoch 
        # loading checkpoint
        print("loading checkpoint: ", test_model_path) #args.test_model_path
        metric_fc = load_test_models(metric_fc, test_model_path) #args.test_model_path
        
        #pprint.pprint(args)
        print("\nstart testing ...")
        test(test_dl, model, metric_fc, args)



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