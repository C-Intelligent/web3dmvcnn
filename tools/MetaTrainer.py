import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import os
from tensorboardX import SummaryWriter
import time
import copy
import torch.optim as optim
from models.MetaMVCNN import MetaMVCNN

class ModelNetTrainer(object):
    def __init__(self, model, train_loader, val_loader, meta_loader, optimizer, loss_fn, \
                 model_name, log_dir, num_views, num_classes):

        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.meta_loader = meta_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.log_dir = log_dir
        self.num_views = num_views

        self.num_classes = num_classes

        self.model.cuda()
        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir)


    def train(self, n_epochs):
        print('=============================== START TRAINING ====================================')
        best_acc = 0
        i_acc = 0
        # 启用 Batch Normalization 和 Dropout
        self.model.train()

        ######################## META LOAD DATA ######################

        # meta_net = copy.deepcopy(self.model)
        # if torch.cuda.is_available():
        #     meta_net.cuda()
        # meta_net.train()

        # # meta_net.load_state_dict(self.model.state_dict())

        # # may need to be updated
        # optlist = list(meta_net.net_1.parameters()) + list(meta_net.net_2.parameters())
        # meta_optimizer = optim.Adam(optlist, lr=1e-3, weight_decay=0.001 , betas=(0.9, 0.999))

        ######################## META LOAD DATA ######################

        for epoch in range(n_epochs):
            # permute data for mvcnn
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths)/self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[rand_idx[i]*self.num_views:(rand_idx[i]+1)*self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            # writer.add_scalar()  将标量添加到 summary
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):
                if i % 100 == 0:
                    print('batch ', i)
                ###################### TRAINING SET ########################
                if self.model_name == 'mvcnn':
                    N,V,C,H,W = data[1].size()
                    in_data = Variable(data[1]).view(-1,C,H,W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()
                ###################### TRAINING SET ########################


                ######################## META TRAINING ######################
                meta_net = copy.deepcopy(self.model) # to solve second time backward
                # meta_net.load_state_dict(self.model.state_dict())
                optlist = list(meta_net.mvcnn.parameters())
                meta_optimizer = optim.Adam(optlist, lr=1e-3, weight_decay=0.001 , betas=(0.9, 0.999))


                T_hat = Variable(torch.zeros(self.num_classes, self.num_classes).cuda(), requires_grad=False)
                
                
                y_f_hat  = meta_net(in_data)

                loss1 = F.cross_entropy(y_f_hat, target, reduce=False)

                # loss = self.loss_fn(y_f_hat, target)
                l_f_meta = torch.sum(loss1)


                meta_net.zero_grad()
                '''
                grads = torch.autograd.grad(l_f_meta, (meta_net.parameters()), create_graph=True)
                meta_net.update_params(1e-3, source_params=grads)
                '''
                ############## replace the two lines above ############
                
                l_f_meta.backward()  #  Trying to backward through the graph a second time
                meta_optimizer.step()
                
                ############## replace the two lines above ############
                

                # y_g_hat = torch.Tensor(0, self.num_classes).cuda().requires_grad_(False)
                # grad_eps = Variable(torch.zeros(self.num_classes, self.num_classes).cuda(), requires_grad=False)
                ###################### VALIDATION SET ########################
                for i, meta_data in enumerate(self.meta_loader):
                    if self.model_name == 'mvcnn':
                        N,V,C,H,W = meta_data[1].size()
                        X_meta_b = Variable(meta_data[1], requires_grad=False).view(-1,C,H,W).cuda().requires_grad_(False)
                    else:
                        X_meta_b = Variable(meta_data[1].cuda(), requires_grad=False).requires_grad_(False)
                    
                    Y_meta_b = Variable(meta_data[0].cuda().long(), requires_grad=False)
                    y_g_hat = meta_net(X_meta_b)
                    l_g_meta_loss = F.cross_entropy(y_g_hat, Y_meta_b)

                    grad_eps = torch.autograd.grad(l_g_meta_loss, meta_net.T, only_inputs=True)[0]

                    T_hat += grad_eps
                ###################### VALIDATION SET ########################
                meta_net.T = torch.clamp(meta_net.T-0.11*T_hat,min=0)
 
                norm_c = torch.sum(meta_net.T, 0)
                for j in range(self.num_classes):
                    if norm_c[j] != 0:
                        meta_net.T[:, j] /= norm_c[j]

                ######################## META TRAINING ######################
                out_data = self.model(in_data)
                ########################## T ###############################
                pre2 = torch.mm(out_data, meta_net.T)
                l_f = torch.sum(F.cross_entropy(pre2,target, reduce=False))
                ########################## T ###############################

                self.optimizer.zero_grad()
                ########################## T ###############################
                l_f.backward()
                self.optimizer.step()
                ########################## T ###############################

                del meta_net
                torch.cuda.empty_cache()
                
            i_acc += i

            # evaluation
            if (epoch+1)%1==0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch+1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch+1)
                self.writer.add_scalar('val/val_loss', loss, epoch+1)

            # save best model
            if val_overall_acc > best_acc:
                best_acc = val_overall_acc
                self.model.save(self.log_dir, epoch)
 
            # adjust learning rate manually
            if epoch > 0 and (epoch+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.log_dir+"/all_scalars.json")
        self.writer.close()

    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0

        # in_data = None
        # out_data = None
        # target = None

        # wrong_class = np.zeros(40)
        # samples_class = np.zeros(40)
        wrong_class = np.zeros(self.num_classes)
        samples_class = np.zeros(self.num_classes)
        all_loss = 0

        self.model.eval()

        avgpool = nn.AvgPool1d(1, 1)

        total_time = 0.0
        total_print_time = 0.0
        all_target = []
        all_pred = []

        for _, data in enumerate(self.val_loader, 0):

            if self.model_name == 'mvcnn':
                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W).cuda()
            else:#'svcnn'
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print('Classes ac: \n', (samples_class-wrong_class)/samples_class)

        print ('Total # of test models: ', all_points)
        
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print ('val mean class acc. : ', val_mean_class_acc)
        print ('val overall acc. : ', val_overall_acc)
        print ('val loss : ', loss)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc

