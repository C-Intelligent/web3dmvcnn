import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
import copy

from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)


name = 'mvcnn'
num_models = 1000
weight_decay = 0.001 
num_views = 12 
cnn_name = 'vgg11'
# cnn_name = 'resnet18'
# cnn_name = 'resnet34'
# cnn_name = 'resnet50'
# cnn_name = 'vgg16'
no_pretraining = False
lr = 5e-5
bs = 8
batchSize = bs
svcnn_bs = 32
# web_shapent55_renamed_0.2   modelnet40_images_new_12x
# train_path = 'web_shapent55_renamed_0.2/*/train'
# val_path = 'web_shapent55_renamed_0.2/*/test'
train_path = 'web_shapent10_toy/*/train'
val_path = 'web_shapent10_toy/*/test'


# _classnames=['airplane','bicycle','camera','dishwasher','jar','microwave','printer','stove',
#             'ashcan','birdhouse','can' ,'display' ,'knife' ,'motorcycle','remote_control','table',
#             'bag','bookshelf','cap','earphone','lamp','mug','rifle','telephone',
#             'basket','bottle','car','faucet','laptop','piano','rocket','tower',
#             'bathtub','bowl','chair','file','loudspeaker','pillow','skateboard','train',
#             'bed','bus','clock','guitar','mailbox','pistol','sofa','vessel',
#             'bench','cabinet','computer_keyboard','helmet','microphone','pot','washer']

_classnames=['airplane' , 'bag' , 'bathtub' , 'bench' ,'birdhouse' , 'bottle' , 'bus' , 'camera' , 'cap' , 'chair']
_num_classes = len(_classnames)

pretraining = not no_pretraining
log_dir = name
# create_folder(name)
# config_f = open(os.path.join(log_dir, 'config.json'), 'w')
# json.dump(vars(args), config_f)
# config_f.close()

train_stage1 = False
train_stage2 = True

import time
time_stamp = time.localtime()
_date_str = time.strftime('%y_%m_%d', time_stamp)


# STAGE 2
n_models_train = num_models*num_views

mvcnn_param_path = '/home/cz/src/mvcnn_pytorch/mvcnn_stage_1_0211/'
mvcnn_model_file = 'model-00032.pth'

if train_stage2:
    log_dir = name+'_stage__' + _date_str
    create_folder(log_dir)
    cnet_2 = MVCNN(name, 0, nclasses=_num_classes, cnn_name=cnn_name, num_views=num_views)

    
    # Adam Optimizer是对SGD的扩展，可以代替经典的随机梯度下降法来更有效地更新网络权重。
    optimizer = optim.Adam(cnet_2.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

    train_dataset = MultiviewImgDataset(train_path, classes=copy.deepcopy(_classnames), scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(val_path, classes=copy.deepcopy(_classnames), scale_aug=False, rot_aug=False, num_views=num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = ModelNetTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=num_views, num_classes=_num_classes)

    # Load the history traning weights
    # trainer.model.load(mvcnn_param_path, mvcnn_model_file)

    trainer.train(30)
