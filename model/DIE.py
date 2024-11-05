import torch
import torch.nn as nn
import torch.nn.functional as F  

 


class DIE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.toplayer = nn.Conv2d(1024, dim, 1, 1, 0)
        self.layer1 = nn.Conv2d(512, dim, 1, 1, 0)
        self.layer2 = nn.Conv2d(256, dim, 1, 1, 0)
        self.layer3 = nn.Conv2d(128, dim, 1, 1, 0)
        
        self.smooth1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.smooth2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.smooth3 = nn.Conv2d(dim, dim, 3, 1, 1)
        
        
        self.down1 = nn.Conv2d(dim, dim, 3, 2, 1)
        self.down2 = nn.Conv2d(dim, dim, 3, 2, 1)
        self.down3 = nn.Conv2d(dim, dim, 3, 2, 1)
        
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        p4 = self.toplayer(x[3])
        p3 = self.layer1(x[2])
        p2 = self.layer2(x[1])
        p1 = self.layer3(x[0])
        
        ##
        t = self.down1(p1)
        x2 = t + p2
        
        
        t = self.down2(p2)
        s = self.shortcut1(p1)
        x3 = p3 + t + s
        
        t = self.down3(p3)
        s = self.shortcut2(p1)
        s2 = self.shortcut3(p2)
        x4 = p4 + t + s + s2
        
        ##
        
        
        x2 = self.smooth1(x2)
        
        x3 = self.smooth2(x3)
        
        x4 = self.smooth3(x4)
        
        
        return p1, x2, x3, x4
       