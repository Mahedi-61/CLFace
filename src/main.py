"""
RUN THE CODE
# for base
python3 src/main.py --train --is_base --arch 50 --setup finetune 

# for clface or finetune
python3 src/main.py --train --is_base --add_ckd --setup clface --arch 50 --step_size 5 --base_fraction 0.50 --saved_model_path 50_arcface_50_ms1m_step5.pth
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
from train import CLFace
from LwM import LwM
from eval_tinyface import evaluate_tinyface
from eval_ijb import evaluate_ijb


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',       dest="train",       help='training', action='store_true')
    parser.add_argument('--is_base',     dest="is_base",     help='considering a base step',   action='store_true')
    parser.add_argument('--add_id_loss', dest="add_id_loss", help='considering a base step',   action='store_true')
    parser.add_argument('--add_ckd',     dest="add_ckd",     help='considering a base step',   action='store_true')

    parser.set_defaults(train=False)
    parser.set_defaults(is_base=False)
    parser.set_defaults(add_id_loss=False)
    parser.set_defaults(add_ckd=False)
    parser.set_defaults(lwm=False)

    parser.add_argument('--step',          type=int,   default=1,          help="incremental current step")
    parser.add_argument('--step_size',     type=int,   default=5,          help='total incremental steps')
    parser.add_argument('--freeze',        type=int,   default=0,          help='Number of epoch pretrained model frezees')
    parser.add_argument('--setup',         type=str,   default="clface",   help='finetune | clface | lwm')
    parser.add_argument('--base_fraction', type=float, default=0.50,       help='base fraction 0.10 | 0.25 | 0.50 | 0.75')

    parser.add_argument('--dataset',       type=str,   default="ms1m",    help='Name of the datasets ms1m | vgg | WF12M')
    parser.add_argument("--test_dataset",  type=str,   default="ms1m_test",   help="Name of the test dataset")
    parser.add_argument('--batch_size',    type=int,   default=256,        help='Batch size') #256
    parser.add_argument('--max_epoch',     type=int,   default=9,          help='Maximum epochs') #20
    parser.add_argument('--model_type',    type=str,   default="arcface",  help='architecture of the model: arcface | adaface')
    parser.add_argument('--arch',          type=str,   default="50",       help='architecture of the backend')

    parser.add_argument('--test_file',     type=str,   default="test_pairs.txt",          help='Name of the test list file')
    parser.add_argument('--valid_file',    type=str,   default="valid_pairs.txt",         help='Name of the test list file')
    
    parser.add_argument('--checkpoint_path',    type=str,   default="./checkpoints",       			help='checkpoints directory')
    parser.add_argument('--saved_model_path',   type=str,   default="50_arcface_50_WF12M_step0.pth", help='model pretrained on previous step')

    parser.add_argument('--final_dim',     type=int,     default=512,    help='weight value of the ITC loss')
    parser.add_argument('--s',             type=float,   default=64.0,   help='arcface s')
    parser.add_argument('--m',             type=float,   default=0.5,    help='arcface margin') 
    parser.add_argument('--gpu_id',        type=int,     default=0,      help='GPU ID') 
    return  parser.parse_args(argv)


setup_cfg = SimpleNamespace(
    lr_train= 0.1,   
    weight_decay= 0.0005, 
    momentum= 0.9,
    delta_gpkd  = 12,
    delta_msd = 3,
    delta_ckd = 1,
    num_workers = 4,

    temperature= 2,
    is_roc= False,
    is_ident= False,
    manual_seed = 61,
)


if __name__ == "__main__":
	c_args = parse_arguments(sys.argv[1:])
	args = SimpleNamespace(**c_args.__dict__, **setup_cfg.__dict__)

	if args.dataset == "ms1m": 
		args.num_classes = 85742
	elif args.dataset == "vgg": 
		args.num_classes = 8631
	elif args.dataset == "WF12M": 
		args.num_classes = 600000

	print("********** Dataset Name: %s ***********" % args.dataset)
	random.seed(args.manual_seed)
	np.random.seed(args.manual_seed)
	torch.manual_seed(args.manual_seed)

	torch.cuda.manual_seed_all(args.manual_seed)
	pprint(args)
	val_acc = {}

	if args.train == True:
		if args.is_base == False:
			for i in range(1, args.step_size + 1):
				print("\n\n ******************* Step %d ******************* " % i)
				args.step = i
				args.saved_model_path = "50_%s_%s_%s_step%d.pth" % (args.model_type, args.arch, args.dataset, (i - 1))

				folder = args.saved_model_path.split("_")[-1].replace(".pth", "")
				args.model_path = os.path.join(args.checkpoint_path, 
											args.dataset, 
											args.setup, 
											str(args.step_size), 
											folder, 
											args.saved_model_path)

				if args.setup == "finetune": t = BaseTrainer(args)
				elif args.setup == "clface": t = CLFace(args)
				elif args.setup == "lwm": t = LwM(args)
				val_acc["step%d" % i] = t.train()

			print("Best accuracies in all steps: ")
			print(val_acc)

		else:
			folder = args.saved_model_path.split("_")[-1].replace(".pth", "")
			args.model_path = os.path.join(args.checkpoint_path, 
										args.dataset, 
										args.setup, 
										str(args.step_size), 
										folder, 
										args.saved_model_path)
			if args.setup == "finetune": t = BaseTrainer(args)
			t.train()

	elif args.train == False:
		for dataset in ['ijb-c']: #'lfw', 'agedb', 'calfw', 'cfp_fp',  'cpl_fw'
			args.test_dataset = dataset

			folder = args.saved_model_path.split("_")[-1].replace(".pth", "")
			args.model_path = os.path.join(args.checkpoint_path, 
							            args.dataset, 
							            args.setup, 
							            str(args.step_size), 
							            folder, 
							            args.saved_model_path)

			if dataset in ["tinyface", "ijb-b", "ijb-c"]:
				if dataset == "tinyface":
					evaluate_tinyface(args)

				elif dataset == "ijb-b":
					args.target = "IJBB"
					evaluate_ijb(args)

				elif dataset == "ijb-c":
					args.target = "IJBC"
					evaluate_ijb(args)
			else:
				if args.setup == "finetune": 
					t = BaseTrainer(args)
					val_acc[dataset] = t.test()

				elif args.setup == "clface": 
						t = CLFace(args)
						val_acc[dataset] = t.test()

				elif args.setup == "lwm": 
					t = LwM(args)
					val_acc[dataset] = t.test()

			print("Best accuracies in all datasets ")
			print(val_acc)