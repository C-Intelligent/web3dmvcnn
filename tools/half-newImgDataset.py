import numpy as np
import glob
import torch.utils.data
import os
import math
from skimage import io, transform
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
from torch import tensor
import random

class MyImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True, compute_mean_std=0):
        self.classnames=['airplane','bicycle','camera','dishwasher','jar','microwave','printer','stove',
            'ashcan','birdhouse','can' ,'display' ,'knife' ,'motorcycle','remote_control','table',
            'bag','bookshelf','cap','earphone','lamp','mug','rifle','telephone',
            'basket','bottle','car','faucet','laptop','piano','rocket','tower',
            'bathtub','bowl','chair','file','loudspeaker','pillow','skateboard','train',
            'bed','bus','clock','guitar','mailbox','pistol','sofa','vessel',
            'bench','cabinet','computer_keyboard','helmet','microphone','pot','washer']

        self.num_views = num_views
        
        self.root_dir = root_dir
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode

        self.filepaths = []
        
        
        # compute mean std
        self.means = [tensor(0.)]*3
        self.stdevs = [tensor(0.)]*3
        if 2 == compute_mean_std:
            self.means = [0.81662318, 0.81662318, 0.81662318]
            self.stdevs = [0.24117189, 0.24117189, 0.24117189]
        else:
            self.means = [0.485, 0.456, 0.406]
            self.stdevs = [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.means[0], self.means[1], self.means[2]]
                , std=[self.stdevs[0], self.stdevs[1], self.stdevs[2]])
        ])


    def __len__(self):
        return len(self.filepaths)


class SingleImgDataset(MyImgDataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True):
        MyImgDataset.__init__(self, root_dir, scale_aug, rot_aug, test_mode, num_models, num_views, shuffle, 2)

        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*shaded*.png'))
            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])
        
    
    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)

        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return (class_id, im, path)
    def __len__(self):
        return len(self.filepaths)

class MultiviewImgDataset(MyImgDataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, \
                 num_models=0, num_views=12, shuffle=True):
        MyImgDataset.__init__(self, root_dir, scale_aug, rot_aug, test_mode, num_models, num_views, shuffle, 2)
        
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/',2)[0]
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            ## Select subset for different number of views
            stride = int(12/self.num_views) # 12 6 4 3 2 1
            all_files = all_files[::stride]

            if num_models == 0:
                # Use the whole dataset
                self.filepaths.extend(all_files)
            else:
                self.filepaths.extend(all_files[:min(num_models,len(all_files))])

        if shuffle==True:
            # permute
            rand_idx = np.random.permutation(int(len(self.filepaths)/num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i]*num_views:(rand_idx[i]+1)*num_views])
            self.filepaths = filepaths_new

        if self.test_mode:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        else:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])

    def __len__(self):
        return len(self.filepaths / self.num_views)