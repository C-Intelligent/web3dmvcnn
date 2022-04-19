from models.MVCNN import MVCNN
import torch.nn as nn
import torch
from torch.autograd import Variable
from .Model import Model

class MetaMVCNN(Model):
    def __init__(self, name, nclasses, cnn_name='vgg11', num_views=12, pretrained=True):
        super(MetaMVCNN, self).__init__(name)
        self.mvcnn = MVCNN(name, 0, nclasses, cnn_name, num_views, pretrained)
        self.T = Variable(torch.eye(nclasses, nclasses).cuda(), requires_grad=True)

    def forward(self, x):
        y = self.mvcnn(x)
        return torch.mm(y, self.T) 