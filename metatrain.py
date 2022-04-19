import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse
import copy

from tools.MetaTrainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MetaMVCNN import MetaMVCNN

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
no_pretraining = True
lr = 5e-5
bs = 2
batchSize = bs
svcnn_bs = 32
# web_shapent55_renamed_0.2   modelnet40_images_new_12x
# train_path = 'web_shapent55_renamed_0.2/*/train'
# val_path = 'web_shapent55_renamed_0.2/*/test'
train_path = 'web_shapent10_toy/*/train'
val_path = 'web_shapent10_toy/*/test'

pretraining = not no_pretraining
log_dir = name

import time
time_stamp = time.localtime()
_date_str = time.strftime('%y_%m_%d', time_stamp)


n_models_train = num_models*num_views

# _classnames=['airplane','bicycle','camera','dishwasher','jar','microwave','printer','stove',
#             'ashcan','birdhouse','can' ,'display' ,'knife' ,'motorcycle','remote_control','table',
#             'bag','bookshelf','cap','earphone','lamp','mug','rifle','telephone',
#             'basket','bottle','car','faucet','laptop','piano','rocket','tower',
#             'bathtub','bowl','chair','file','loudspeaker','pillow','skateboard','train',
#             'bed','bus','clock','guitar','mailbox','pistol','sofa','vessel',
#             'bench','cabinet','computer_keyboard','helmet','microphone','pot','washer']

_classnames=['airplane' , 'bag' , 'bathtub' , 'bench' ,'birdhouse' , 'bottle' , 'bus' , 'camera' , 'cap' , 'chair']
_num_classes = len(_classnames)

# torch.cuda.set_device(1)

log_dir = name+'_stage_meta_' + _date_str
create_folder(log_dir)
meta_main = MetaMVCNN(name, nclasses=_num_classes, cnn_name=cnn_name, num_views=num_views)

# Load the history traning weights

# Adam Optimizer是对SGD的扩展，可以代替经典的随机梯度下降法来更有效地更新网络权重。

optlist = list(meta_main.mvcnn.parameters())
optimizer = optim.Adam(optlist, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))

train_dataset = MultiviewImgDataset(train_path, classes=copy.deepcopy(_classnames), scale_aug=False, rot_aug=False, num_models=n_models_train, num_views=num_views)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=False, num_workers=0)# shuffle needs to be false! it's done within the trainer

meta_num = 50
meta_val_dataset = MultiviewImgDataset(val_path, classes=copy.deepcopy(_classnames), scale_aug=False, rot_aug=False, num_views=num_views)
meta_val_dataset.val_set_select(meta_num)
meta_loader = torch.utils.data.DataLoader(meta_val_dataset, batch_size=batchSize, shuffle=False, num_workers=0)

val_dataset = MultiviewImgDataset(val_path, classes=copy.deepcopy(_classnames), scale_aug=False, rot_aug=False, num_views=num_views)
val_dataset.eliminate(meta_val_dataset.filepaths)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchSize, shuffle=False, num_workers=0)


'''
V,C,H,W = meta_val_dataset[0][1].size()
meta_data_x = torch.Tensor(meta_num, V,C,H,W)
meta_data_y = torch.Tensor(meta_num)
for i, m in enumerate(meta_val_dataset):
    meta_data_y[i] = m[0]
    meta_data_x[i] = m[1]

'''

print('num_train_files: '+str(len(train_dataset.filepaths)))
print('num_val_files: '+str(len(val_dataset.filepaths)))
print('num_meta_val_files: '+str(len(meta_val_dataset.filepaths)))
print('batchs: ', len(train_dataset.filepaths) / (batchSize * num_views))

trainer = ModelNetTrainer(meta_main, train_loader, val_loader, meta_loader, optimizer, nn.CrossEntropyLoss(), 'mvcnn', log_dir, num_views=num_views, num_classes=_num_classes)

param_path = '/home/cz/src/mvcnn_pytorch/mvcnn_stage_meta_22_03_23/'
model_file = 'model-00006.pth'

if pretraining:
    trainer.model.load(param_path, model_file)

trainer.train(30)
