import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import os
import torch

import numpy as np

class TTRCU(nn.Module):
    def __init__(self, input_size, output_size, device='cpu'):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.PReLU(),
            nn.Linear(512,512),
            nn.PReLU(),
            nn.Linear(512,512),
            nn.PReLU(),
            nn.Linear(512,512),
            nn.PReLU(),
            nn.Linear(512,output_size)
        )
        self.device=device
    def forward(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, device=self.device, dtype=torch.float)
        s = s.view(s.shape[0],-1)
        logits = self.model(s)
        return logits.view(logits.shape[0])


class EstimatorModel(nn.Module):
    
    def __init__(self):
        
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
        
        #y = self.sigmoid(y)
                
        return y

class ClassifierModel(nn.Module):
    
    def __init__(self):
        
        super(ClassifierModel, self).__init__()
        
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