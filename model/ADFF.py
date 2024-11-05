import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SIE import SIE
from model.DIE import DIE
       



class ADFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.die0 = DIE(256)
        self.die1 = DIE(256)
        
       
        self.sie = SIE(256)
        
        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True)
        )
        self.calories = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.mass = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.fat = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.carb = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        self.protein = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1))
        
    def forward(self, rgb, rgbd):
        
               
        rgb = self.die0(rgb)
        rgbd = self.die1(rgbd)
        
        x1, x2, x3, x4 = self.sie(rgb, rgbd)
        
        x1 = self.GAP(x1)
        x2 = self.GAP(x2)
        x3 = self.GAP(x3)
        x4 = self.GAP(x4)
        
        
        
        inputs = torch.cat([x1, x2, x3, x4], 1)
        inputs = inputs.view(inputs.size(0), -1)
        inputs = self.fc(inputs)
        
        results = []
        results.append(self.calories(inputs).squeeze())
        results.append(self.mass(inputs).squeeze())
        results.append(self.fat(inputs).squeeze())
        results.append(self.carb(inputs).squeeze())
        results.append(self.protein(inputs).squeeze())
        
        
        return results

 





        
        
        

        
   
   
    
    
def makeweight(num):
    params = torch.ones(num, requires_grad=True)
    w = torch.nn.Parameter(params)
        
    return w
    
def weightm(x, weight):
    out = []
    for i in range(len(x)):
        t = weight[i] * x[i]
        out.append(t)
        
    t = torch.cat(out, dim=1)
    out = t.view(t.size(0), -1)
       
    return out