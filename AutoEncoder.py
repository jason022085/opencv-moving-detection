# -*- coding: utf-8 -*-
import torch 
from torch import nn
import cv2     
import numpy as np
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 =nn.Sequential(
                nn.Conv2d(3,32, kernel_size = 5 , padding=2 , stride=2 ),
                #nn.MaxPool2d(3,stride = 2)
                )
        self.resBlock1 = nn.Sequential(
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32,32, kernel_size = 3 , padding=1 , stride=2 ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32,32, kernel_size = 3 , padding=1 , stride=1 ),
                )
        self.shortCut1 = nn.Conv2d(32,32, kernel_size = 1 , padding=0 , stride=2)
        
        self.resBlock2 = nn.Sequential(
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32,64, kernel_size = 3 , padding=1 , stride=2 ),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,64, kernel_size = 3 , padding=1 , stride=1 ),
                )
        self.shortCut2 = nn.Conv2d(32,64, kernel_size = 1 , padding=0 , stride=2)
        
        self.resBlock3 = nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64,128, kernel_size = 3 , padding=1 , stride=2 ),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128,128, kernel_size = 3 , padding=1 , stride=1 ),
                )
        self.shortCut3 = nn.Conv2d(64,128, kernel_size = 1 , padding=0 , stride=2)
        self.Trans1 = nn.Sequential(
                nn.ConvTranspose2d(128,64,kernel_size=[2,2],stride=2),
                nn.ReLU(),
                nn.Conv2d(64,64, kernel_size = 3 , padding=1 , stride=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64,32,kernel_size=[2,2],stride=2),
                nn.ReLU(),
                nn.Conv2d(32,32, kernel_size = 3 , padding=1 , stride=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32,16,kernel_size=[2,2],stride=2),
                nn.ReLU(),
                nn.Conv2d(16,16, kernel_size = 3 , padding=1 , stride=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16,8,kernel_size=[2,2],stride=2),
                nn.ReLU(),
                nn.Conv2d(8,4, kernel_size = 3 , padding=1 , stride=1),
                nn.ReLU(),
                nn.Conv2d(4,3, kernel_size = 3 , padding=1 , stride=1),
                #nn.ReLU()
                ) 
        self.codeConvert = nn.Sequential(
                nn.Linear(16384*2,1024),
                nn.ReLU(),
                nn.Linear(1024,16384*2),
                nn.ReLU(),
            )
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        sc = self.shortCut1(x)
        x = self.resBlock1(x)
        x = x+sc
        sc = self.shortCut2(x)
        x = self.resBlock2(x)
        x = x+sc
        sc = self.shortCut3(x)
        x = self.resBlock3(x)
        x = x+sc
        fixShape = x.shape
        x = x.view(x.shape[0],-1)
        x = self.codeConvert(x)    
        x = x.view(fixShape)
        x = self.Trans1(x)
        return torch.sigmoid(x)