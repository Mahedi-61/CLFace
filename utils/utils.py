import os
import errno
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
import datetime
import dateutil.tz
from PIL import Image 
from albumentations.pytorch import ToTensorV2
import albumentations as A 
from torchvision import transforms

# config
def get_time_stamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')  
    return timestamp


def load_yaml(filename):
    with open(filename, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg


def merge_args_yaml(args):
    if args.cfg_file is not None:
        opt = vars(args)
        args = load_yaml(args.cfg_file)
        args.update(opt)
        args = edict(args)
    return args



def get_transform_img(img_path, split, model_type="arcface"):
    img = np.array(Image.open(img_path).convert('RGB')) 

    train_transforms = A.Compose([
        A.HorizontalFlip(p = 0.5),
        A.Normalize(mean=torch.tensor([0.5412688 , 0.43232402, 0.37956172]), 
                    std=torch.tensor([0.28520286, 0.2531577 , 0.24701026]),
                    always_apply=True),
        ToTensorV2()
    ])

    valid_transforms = A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
        ToTensorV2()
    ])

    if split == "train": tfms = train_transforms
    elif split == "test" or split == "valid":  tfms = valid_transforms

    img = tfms(image=img)["image"] 
    if model_type == "adaface": 
        permute = [2, 1, 0]
        img = img[permute, :, :] #RGB --> BGR

    return img 


def do_flip_test_images(img_path, model_type="arcface"):
    img = np.array(Image.open(img_path).convert('RGB')) 
    tfms = A.Compose([
        A.HorizontalFlip(p = 1),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
        ToTensorV2()
    ])

    img = tfms(image=img)["image"] 
    if model_type == "adaface": 
        permute = [2, 1, 0]
        img = img[permute, :, :] #RGB --> BGR
    return img


def get_imgs_dir(data_dir, args):
    img_dir = os.path.join(data_dir, "images")
    sub_ls = sorted(os.listdir(img_dir), key= lambda x: int(x))
    
    if args.step == 0:
        sub_ls = sub_ls[ : args.base_num]
    else:
        begin_index = args.base_num + (args.class_range * (args.step - 1))
        end_index = begin_index + args.class_range
        sub_ls = sub_ls[begin_index : end_index]

    all_imgs = []
    all_labels = []

    print("base number: ", args.base_num)
    print("class_range: ", args.class_range)
    print("Step: %d | Subject list: %s, %s ..... %s, %s" % 
          (args.step, sub_ls[0], sub_ls[1], sub_ls[-2], sub_ls[-1]))
    
    for sub in sub_ls:
        sub_dir = os.path.join(img_dir, sub)
        #sub_imgs = sorted(os.listdir(sub_dir), key= lambda x: int((x.split("_")[-1]).split(".")[0]))

        sub_imgs_dir = [os.path.join(sub_dir, img) for img in os.listdir(sub_dir)]
        all_imgs += sub_imgs_dir

        if args.step == 0:
            all_labels += [int(sub)] * len(sub_imgs_dir) 
        else:
            all_labels += [int(sub) - args.base_num] * len(sub_imgs_dir) 

    if args.step == 0:
        print("Labels: %d, %d ..... %d, %d" % 
          (int(sub_ls[0]), int(sub_ls[1]), int(sub_ls[-2]), int(sub_ls[-1])))
    else:
        print("Labels: %d, %d ..... %d, %d" % 
          (int(sub_ls[0]) - args.base_num, int(sub_ls[1]) - args.base_num, 
           int(sub_ls[-2]) - args.base_num, int(sub_ls[-1]) - args.base_num))
    
    print("Total images: ", len(all_imgs))
    return all_imgs, all_labels


def load_model_weights(model, weights, multi_gpus = False):
    if list(weights.keys())[0].find('module')==-1:
        pretrained_with_multi_gpu = False
    else:
        pretrained_with_multi_gpu = True

    if (multi_gpus==False):
        if pretrained_with_multi_gpu:
            state_dict = {
                key[7:]: value
                for key, value in weights.items()
            }
        else:
            state_dict = weights
    else:
        state_dict = weights
    model.load_state_dict(state_dict)
    return model


def load_base_model(model, metric_fc, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['base_model'])
    metric_fc.load_state_dict(checkpoint['metric_Fc'])
    return model, metric_fc


def load_full_model(model, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = load_model_weights(model, checkpoint['model']['model'])
    return model  


def load_shared_model(shared_net, model_path):
    model_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    shared_net = load_model_weights(shared_net, model_checkpoint['model']['shared_net'])
    return shared_net 


if __name__ == "__main__":
    from easydict import EasyDict
    args = EasyDict()
    args.step = 5
    args.class_range= 8574
    data_dir = "./data/ms1m"
    get_imgs_dir(data_dir, args)