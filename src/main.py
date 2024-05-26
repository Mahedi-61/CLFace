"""
RUN THE CODE
# for base
python3 src/main.py --train --is_base --step 0 --setup finetune --test_dataset lfw

# for clface or finetune
python3 src/main.py --train --step 1 --setup finetune --test_dataset lfw --saved_model_path arcface_18_ms1m_step0.pth
"""

import os, sys, random
import os.path as osp
import argparse
import torch
import numpy as np
from pretty_simple_namespace import pprint
import torch

from types import SimpleNamespace
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from base_train import BaseTrainer 
from train1 import CLFace


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',       dest="train",      help='training',                  action='store_true')
    parser.add_argument('--is_base',     dest="is_base",    help='considering a base step',   action='store_true')
    parser.set_defaults(train=False)
    parser.set_defaults(is_base=False)

    parser.add_argument('--step',          type=int,   default=1,         help="incremental current step")
    parser.add_argument('--num_classes',   type=int,   default=85_742,    help='number of classes')
    parser.add_argument('--step_size',     type=int,   default=10,         help='total incremental steps')
    parser.add_argument('--freeze',        type=int,   default=3,          help='Number of epoch pretrained model frezees')
    parser.add_argument('--setup',         type=str,   default="finetune",   help='finetune | clface')

    parser.add_argument('--dataset',       type=str,   default="ms1m",       help='Name of the datasets ms1m')
    parser.add_argument("--test_dataset",  type=str,   default="lfw",      help="Name of the test datasets")
    parser.add_argument('--batch_size',    type=int,   default=256,          help='Batch size')
    parser.add_argument('--max_epoch',     type=int,   default=12,                      help='Maximum epochs') #20
    parser.add_argument('--model_type',    type=str,   default="arcface",               help='architecture of the model: arcface | adaface')
    parser.add_argument('--test_file',     type=str,   default="test_pairs.txt",          help='Name of the test list file')
    parser.add_argument('--valid_file',    type=str,   default="valid_pairs.txt",         help='Name of the test list file')
    
    parser.add_argument('--checkpoint_path',    type=str,   default="./checkpoints",       help='checkpoints directory')
    parser.add_argument('--saved_model_path',   type=str,   default="arcface_18_ms1m_step0.pth", help='model pretrained on base/previous step data')

    parser.add_argument('--final_dim',     type=int,   default=512,    help='weight value of the ITC loss')
    parser.add_argument('--s',             type=float,   default=64.0,    help='weight value of the attribute loss')
    parser.add_argument('--m',             type=float,   default=0.5,   help='weight value of the KD loss') 
    return  parser.parse_args(argv)



setup_cfg = SimpleNamespace(
    lr_train= 0.1,   
    weight_decay= 0.0005, 
    momentum= 0.9,
    delta_gpkd  = 2,
    delta_pod = 1,
    kl_div= 550,
    num_workers = 4,

    temperature= 2,
    is_roc= False,
    is_ident= False,
    use_se = False,  

    num_imposter_per_sub=1,
    manual_seed = 61
)


if __name__ == "__main__":
    c_args = parse_arguments(sys.argv[1:])

    if c_args.dataset == "ms1m":
        args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__)

    print("********** Dataset Name: %s ***********" % args.dataset)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    torch.cuda.manual_seed_all(args.manual_seed)
    pprint(args)
    
    if args.train == True:
        for i in range(1, args.step_size + 1):
            print("\n\n ******************* Step %d ******************* " % i)
            args.step = i
            args.saved_model_path = "arcface_18_ms1m_step%d.pth" % (i - 1)

            folder = args.saved_model_path.split("_")[-1].replace(".pth", "")
            args.model_path = os.path.join(args.checkpoint_path, 
                                        args.dataset, 
                                        args.setup, 
                                        str(args.step_size), 
                                        folder, 
                                        args.saved_model_path)
            
            if args.setup == "finetune": t = BaseTrainer(args)
            elif args.setup == "clface": t = CLFace(args)
            t.train()

    elif args.train == False:
        for dataset in ['lfw', 'agedb', 'calfw', 'cpl_fw', 'cfp_fp']:
            args.test_dataset = dataset

            folder = args.saved_model_path.split("_")[-1].replace(".pth", "")
            args.model_path = os.path.join(args.checkpoint_path, 
                                        args.dataset, 
                                        args.setup, 
                                        str(args.step_size), 
                                        folder, 
                                        args.saved_model_path)

            if args.setup == "finetune": t = BaseTrainer(args)
            elif args.setup == "clface": t = CLFace(args)
            t.test()