import torch
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from PIL import Image


def get_imgs_dir(data_dir, split, step, args):
    split_dir = os.path.join(data_dir, split)
    sub_ls = sorted(os.listdir(split_dir), key= lambda x: int(x))

    all_imgs = []
    all_labels = []
    sub_ls = get_step_subs(sub_ls, step, args.class_range)
    print(sub_ls)

    for sub in sub_ls:
        sub_dir = os.path.join(split_dir, sub)
        sub_imgs = sorted(os.listdir(sub_dir), key= lambda x: int((x.split("_")[-1]).split(".")[0]))

        sub_imgs_dir = [os.path.join(sub_dir, img) for img in sub_imgs]

        all_imgs += sub_imgs_dir
        all_labels += [int(sub)] * len(sub_imgs_dir) #- (step * args.class_range)

    return all_imgs, all_labels


def get_step_subs(sub_ls, step, class_range):
    begin_index = class_range * step
    end_index = begin_index + class_range
    return sub_ls[begin_index : end_index]


def get_imgs(img_path, transform=None, model_type="arcface"):

    img = Image.open(img_path).convert('RGB')
    norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if transform is not None:
        img = transform(img)
    else:
        img = norm(img)
        
    if model_type == "adaface": 
        permute = [2, 1, 0]
        img = img[permute, :, :] #RGB --> BGR
    return img


if __name__ == "__main__":
    from easydict import EasyDict
    args = EasyDict()
    step = 2
    args.class_range= 50
    data_dir = "./data/facescrub"
    split = "valid"
    get_imgs_dir(data_dir, split, args)