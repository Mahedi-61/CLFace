import os, sys, random
import os.path as osp
import argparse, itertools
import torch  

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from utils.utils import *

################################################################
#                    Train Dataset
################################################################

class TrainDataset():
    def __init__(self, args):
        print("\nLoading %s dataset: " % args.split)
        self.data_dir = os.path.join("./data", args.dataset)
        self.model_type = args.model_type
        self.filenames, self.class_id = get_imgs_dir(self.data_dir, args)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_name = self.filenames[index]
        cls_id = self.class_id[index]

        imgs = get_transform_img(img_name, "train", self.model_type)
        return imgs, torch.tensor(cls_id) 


class TestDataset:
    def __init__(self, args):
        self.data_dir = os.path.join("./data", args.test_dataset)
        self.model_type = args.model_type 

        print("Loading %s dataset: %s" % (args.split, args.test_dataset))
        self.imgs_pair, self.pair_label = self.get_test_list(args.ver_list)


    def get_test_list(self, test_ver_list):
        with open(test_ver_list, 'r') as fd:
            pairs = fd.readlines()
        imgs_pair = []
        pair_label = []

        for pair in pairs:
            splits = pair.split(" ")
            imgs = [splits[0], splits[1]]
            imgs_pair.append(imgs)
            pair_label.append(int(splits[2]))
        return imgs_pair, pair_label


    def __getitem__(self, index):
        imgs = self.imgs_pair[index]
        pair_label = self.pair_label[index]

        data_dir = os.path.join(self.data_dir, "images")
        img1_path = os.path.join(data_dir, "test", imgs[0])
        img2_path = os.path.join(data_dir, "test",  imgs[1])

        img1 = get_transform_img(img1_path, "test", self.model_type)
        img2 = get_transform_img(img2_path, "test", self.model_type)

        img1_h = do_flip_test_images(img1_path, self.model_type)
        img2_h = do_flip_test_images(img2_path, self.model_type)

        return img1, img2, img1_h, img2_h, pair_label


    def __len__(self):
        return len (self.imgs_pair)
    

