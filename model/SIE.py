import torch
import torch.nn as nn
import torch.nn.functional as F   
 
class SIE(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.f1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.f2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.f3 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.f4 = nn.Conv2d(channel, channel, 3, 1, 1)
        
        self.aw1 = self.makeweight(2)
        self.aw4 = self.makeweight(2)
        self.aw2 = self.makeweight(2)
        self.aw3 = self.makeweight(2)
             
        
        
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.c1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.c2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.c3 = nn.Conv2d(channel, channel, 3, 1, 1)
       
        
        self.smooth1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.smooth2 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.smooth3 = nn.Conv2d(channel, channel, 3, 1, 1)
        
        
        self.relu = nn.ReLU()
    def forward(self, rgb, rgbd):
        
        aw1 = self.relu(self.aw1)
        aw2 = self.relu(self.aw2)
        aw3 = self.relu(self.aw3)
        aw4 = self.relu(self.aw4)
        
        x1 = self.weightadd(2, [rgb[0], rgbd[0]], aw1)
        x2 = self.weightadd(2, [rgb[1], rgbd[1]], aw2)
        x3 = self.weightadd(2, [rgb[2], rgbd[2]], aw3)
        x4 = self.weightadd(2, [rgb[3], rgbd[3]], aw4)
        
        
        res1 = self.f1(x1)
        x1 = self.relu(x1 + res1)
       
        res2 = self.f2(x2)
        x2 = self.relu(x2 + res2)
         
        res3 = self.f3(x3)
        x3 = self.relu(x3 + res3)
        
        res4 = self.f4(x4)
        x4 = self.relu(x4 + res4)
        
        
        
        t = self.up(x4)
        res = self.c3(t)
        t = self.relu(res + t)
        x3 = t + x3
        
        
        t = self.up(x3)
        res = self.c2(t)
        t = self.relu(res + t)
        x2 = t + x2
        
         
        t = self.up(x2)
        res = self.c1(t)
        t = self.relu(res + t)
        x1 = t + x1
        
            
        x1 = self.smooth1(x1)
        x2 = self.smooth2(x2)
        x3 = self.smooth3(x3)
        
        
        
        
        return x1, x2, x3, x4
        
    
    def makeweight(self, num):
        params = torch.ones(num, requires_grad=True)
        w = torch.nn.Parameter(params)
        
        return w
    
    def weightadd(self, num, x, weight):
       
        for i in range(num):
            if i == 0:
                t = weight[i] * x[i]
                y = weight[i]
            else:
                t = t + weight[i] * x[i]
                y = y + weight[i]    
            
        y = y + 1e-5
        out = torch.div(t , y)
        
        return out  
  
        
        