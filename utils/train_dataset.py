import torch.utils.data as data
import os
import numpy as np
import torch  
from utils.dataset_utils import *

################################################################
#                    Train Dataset
################################################################

class FacescrubClsDataset(data.Dataset):
    def __init__(self, transform=None, split="train", step=0, args=None):

        print("\nLoading %s dataset: " % split)
        self.transform = transform
        self.dataset_name = args.dataset_name
        self.model_type = args.model_type
        self.split = split
        self.data_dir = os.path.join(args.data_dir, self.dataset_name)
            
        
        self.filenames, self.class_id = get_imgs_dir(self.data_dir, self.split, step, args)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_name = self.filenames[index]
        cls_id = self.class_id[index]

        imgs = get_imgs(img_name, self.transform, self.model_type)
        return imgs, torch.tensor(cls_id) 
