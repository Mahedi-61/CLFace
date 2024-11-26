import torch
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd

import sys, os
import os.path as osp
ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from models.ir_resnet import iresnet50, iresnet100
from utils import tinyface_helper
from utils.modules import prepare_arcface


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def l2_norm(input, axis=1):
    """l2 normalize
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm


def fuse_features_with_norm(stacked_embeddings, stacked_norms, fusion_method='norm_weighted_avg'):

    assert stacked_embeddings.ndim == 3 # (n_features_to_fuse, batch_size, channel)
    if stacked_norms is not None:
        assert stacked_norms.ndim == 3 # (n_features_to_fuse, batch_size, 1)
    else:
        assert fusion_method not in ['norm_weighted_avg', 'pre_norm_vector_add']

    if fusion_method == 'norm_weighted_avg':
        weights = stacked_norms / stacked_norms.sum(dim=0, keepdim=True)
        fused = (stacked_embeddings * weights).sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        fused_norm = stacked_norms.mean(dim=0)

    elif fusion_method == 'pre_norm_vector_add':
        pre_norm_embeddings = stacked_embeddings * stacked_norms
        fused = pre_norm_embeddings.sum(dim=0)
        fused, fused_norm = l2_norm(fused, axis=1)

    elif fusion_method == 'average':
        fused = stacked_embeddings.sum(dim=0)
        fused, _ = l2_norm(fused, axis=1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)

    elif fusion_method == 'concat':
        fused = torch.cat([stacked_embeddings[0], stacked_embeddings[1]], dim=-1)
        if stacked_norms is None:
            fused_norm = torch.ones((len(fused), 1))
        else:
            fused_norm = stacked_norms.mean(dim=0)

    elif fusion_method == 'faceness_score':
        raise ValueError('not implemented yet. please refer to ...')
        # note that they do not use normalization afterward.
    else:
        raise ValueError('not a correct fusion method', fusion_method)

    return fused, fused_norm


def infer(model, dataloader, use_flip_test, fusion_method):
    model.eval()
    features = []
    norms = []
    with torch.no_grad():
        for images, idx in tqdm(dataloader):

            feature = model(images.to("cuda:0"))
            if isinstance(feature, tuple):
                feature, _, _, _ = feature
                norm = None # norm is not used in this case


            if use_flip_test:
                fliped_images = torch.flip(images, dims=[3])
                flipped_feature = model(fliped_images.to("cuda:0"))
                if isinstance(flipped_feature, tuple):
                    flipped_feature, _, _, _ = flipped_feature
                    flipped_norm = None


                stacked_embeddings = torch.stack([feature, flipped_feature], dim=0)
                if norm is not None:
                    stacked_norms = torch.stack([norm, flipped_norm], dim=0)
                else:
                    stacked_norms = None

                fused_feature, fused_norm = fuse_features_with_norm(stacked_embeddings, 
                                                stacked_norms, fusion_method=fusion_method)
                features.append(fused_feature.cpu().numpy())
                norms.append(fused_norm.cpu().numpy())
            else:
                features.append(feature.cpu().numpy())
                norms.append(norm.cpu().numpy())

    features = np.concatenate(features, axis=0)
    norms = np.concatenate(norms, axis=0)
    return features, norms


def evaluate_tinyface(args):
    args.data_root = "data/tinyface"
    args.batch_size = 64
    args.fusion_method = "average" # 'norm_weighted_avg', 'pre_norm_vector_add', 'concat', 'faceness_score
    args.use_flip_test = str2bool

    my_device = torch.device('cuda:%d' % args.gpu_id 
                    if torch.cuda.is_available() else 'cpu')

    args.my_device = my_device 
    model = prepare_arcface(args)

    tinyface_test = tinyface_helper.TinyFaceTest(tinyface_root=args.data_root,
                                    alignment_dir_name='align')


    save_path = os.path.join('./tinyface_result', 
                    args.model_type, "fusion_{}".format(args.fusion_method))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('save_path: {}'.format(save_path))

    img_paths = tinyface_test.image_paths
    print('total images : {}'.format(len(img_paths)))

    dataloader = tinyface_helper.prepare_dataloader(img_paths,  
                                            args.batch_size, 
                                            num_workers=0)
    features, norms = infer(model, dataloader, 
                            use_flip_test=args.use_flip_test, 
                            fusion_method=args.fusion_method)

    results = tinyface_test.test_identification(features, ranks=[1, 5, 20])
    print(results)
    #pd.DataFrame({'rank':[1,5,20], 'values':results}).to_csv(os.path.join(save_path, 'result.csv'))