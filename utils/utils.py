import os
import errno
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
import datetime
import dateutil.tz


def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

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


def save_args(save_path, args):
    fp = open(save_path, 'w')
    fp.write(yaml.dump(args))
    fp.close()



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




def save_base_model(model, epoch, args):
    save_dir = os.path.join(args.checkpoints_path, 
                            args.dataset_name, 
                            args.CONFIG_NAME,  
                            args.model_type)
    mkdir_p(save_dir)

    name = '%s_base_epoch_%d.pth' % (args.model_type, epoch)
    state_path = os.path.join(save_dir, name)
    state = {'base_model': model.state_dict()}
    torch.save(state, state_path)
    print("saving ... ", name)


def save_model(metric_fc, epoch, args):
    save_dir = os.path.join(args.checkpoints_path, 
                            args.dataset_name, 
                            args.CONFIG_NAME,  
                            args.model_type)
    mkdir_p(save_dir)

    name = '%s_full_frozen_epoch_%d.pth' % (args.model_type, epoch)
    state_path = os.path.join(save_dir, name)
    state = {'model': {'metric_fc': metric_fc.state_dict()}}
    torch.save(state, state_path)


def save_shared_models(shared_net, metric_fc, epoch, args):
    save_model_dir = os.path.join(args.checkpoints_path, 
                                  args.dataset_name, 
                                  args.CONFIG_NAME,  
                                  args.model_type)
    
    
    mkdir_p(save_model_dir)

    model_name = '%s_shared_epoch_%d.pth' % (args.model_type, epoch)
    state_model_path = os.path.join(save_model_dir, model_name)

    state_model = {"model": {'shared_net': shared_net.state_dict(), 
                             'metric_fc': metric_fc.state_dict()}}

    torch.save(state_model, state_model_path)




def load_base_model(model, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = load_model_weights(model, checkpoint['base_model'])
    return model 


def load_full_frozen_model(metric_fc, path):
    print("loading full frozen model .....")
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    metric_fc = load_model_weights(metric_fc, checkpoint['model']['metric_fc'])
    return metric_fc  


def load_shared_model(shared_net, model_path):
    model_checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    shared_net = load_model_weights(shared_net, model_checkpoint['model']['shared_net'])
    return shared_net 