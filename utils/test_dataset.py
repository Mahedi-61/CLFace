import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np 
import os
import numpy.random as random
from utils.dataset_utils import * 

################################################################
#                    Test Dataset
################################################################
class FacescrubTestDataset(data.Dataset):
    def __init__(self, transform=None, task=0, args=None):
        self.split= "test"
        if args.dataset_name == "lfw": self.split = "valid"
        self.task = task 
        self.transform = transform
        self.dataset_name = args.dataset_name
        self.model_type = args.model_type 
        self.is_ident = args.is_ident 

        self.data_dir = os.path.join(args.data_dir, args.dataset_name)
        self.test_pair_list = os.path.join(self.data_dir, "script", 
                                "%s_ident_task_%d.txt" % (args.dataset_name, task))
        print("loading test file: ", self.test_pair_list)


        self.img_pairs, self.pair_label = self.get_test_list()
        

    def get_test_list(self):
        with open(self.test_pair_list, 'r') as fd:
            pairs = fd.readlines()
        img_pairs = []
        pair_label = []
        for pair in pairs:
            splits = pair.split(" ")
            imgs = [splits[0], splits[1]]
            img_pairs.append(imgs)
            pair_label.append(int(splits[2]))
        return img_pairs, pair_label

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, index):
        imgs = self.img_pairs[index]
        pair_label = self.pair_label[index]

        img1_name = os.path.join(self.data_dir, self.split, 
                            imgs[0].split("_")[0], imgs[0]) #subject_id, img
        img2_name = os.path.join(self.data_dir, self.split, 
                            imgs[1].split("_")[0], imgs[1])

        img1 = get_imgs(img1_name, self.transform, self.model_type)
        img2 = get_imgs(img2_name, self.transform, self.model_type)

        return img1, img2, torch.tensor(pair_label)
