import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import os
import torch

import numpy as np

class EstimatorModel(nn.Module):
    
    def __init__(self, in_dim):
        
        super(EstimatorModel, self).__init__()
        
        """
        self.linear1 = nn.Linear(in_dim, 500)
        self.linear2 = nn.Linear(500, 200)
        self.linear3 = nn.Linear(200, 100)
        self.linear4 = nn.Linear(100, 1)
        self.tanh = nn.Tanh()
        """
        
        self.linear = nn.Linear(67, 1)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(0.5)
        
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, ext):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #(8, 14, 14)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #(16, 7, 7)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #(32, 4, 4)
        
        x = self.conv4(x)
        x = self.relu(x)
        x = self.avgpool(x)
        #(64, 1, 1)
        
        x = torch.flatten(x, 1)
        
        x = torch.cat([x, ext], 1)
        
        y = self.linear(x)
        
        y = self.sigmoid(y)
                
        return y