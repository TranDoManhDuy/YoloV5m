import torch
import torch.nn as nn
from architecture import *
from blockModel import *

class YoloV5_backbone(nn.Module):
    def __init__(self, in_channels, device):
        super(YoloV5_backbone, self).__init__()
        self.device = device
        self.hooks = []
        self.index = 0
        self.in_channels = in_channels
        self.backboneArchitecture = get_backbone_config()
        self.backbone = self._create_backbone()
    def forward(self, x):
        self.hooks = []
        self.index = 0
        for layer in self.backbone:
            x = layer(x)
            self.hooks.append([self.index, x])
            self.index = self.index + 1
        return self.hooks
    
    def _create_backbone(self):
        layers = []
        in_channels = self.in_channels
        for layer in self.backboneArchitecture:
            if layer[2] == 'conv':
                layers += [
                    CBSBlock(
                        in_channels=in_channels,
                        out_channels=layer[3][0],
                        kernel_size=layer[3][1],
                        stride=layer[3][2],
                        padding=layer[3][3]
                    ).to(self.device)
                ]
                in_channels = layer[3][0]
            if layer[2] == 'c3':
                layers += [
                    C3(
                        in_channels=in_channels,
                        out_channels=layer[3][0],
                        n = layer[1],
                        shortcut=True
                    ).to(self.device)
                ]
                in_channels = layer[3][0]
            if layer[2] == 'sppf':
                layers += [
                    SPPF(
                        in_channels=in_channels,
                        out_channels=layer[3][0],
                        kernel_size=layer[3][1]
                    ).to(self.device)
                ]
        return nn.ModuleList(layers)
    
class YoloV5_head(nn.Module):
    def __init__(self, in_channels, device):
        super(YoloV5_head, self).__init__()
        self.device = device
        self.hooks = []
        self.index = 0
        self.in_channels = in_channels
        self.headArchitecture = get_head_config()
        self.head = self._create_head()
        
    def _create_head(self):
        layers = []
        in_channels = self.in_channels
        for layer in self.headArchitecture:
            if layer[2] == "conv":
                layers  += [
                    CBSBlock(
                    in_channels=in_channels,
                    out_channels= layer[3][0],
                    kernel_size=layer[3][1],
                    stride=layer[3][2],
                    padding=layer[3][3]
                    ).to(self.device)
                ]
                in_channels = layer[3][0]
                
            if layer[2] == "upsample":
                layers += [Upsample().to(self.device)]
            
            if layer[2] == "concat":
                layers += [nn.Identity()]
                in_channels = in_channels * 2
            if layer[2] == "c3":
                layers += [
                    C3(
                    in_channels=in_channels,
                    out_channels=layer[3][0],
                    n=layer[1],
                    shortcut=layer[3][1]
                    ).to(self.device)
                ]
                in_channels = layer[3][0]
        return nn.ModuleList(layers)

    def forward(self, x):
        pointer = 0
        for _, layer in enumerate(self.headArchitecture):
            if layer[2] == "conv" or layer[2] == "upsample" or layer[2] == "c3":
                x = self.head[pointer](x)
                self.hooks.append([self.index, x])
                self.index += 1
                pointer += 1
            elif layer[2] == "concat":
                featureBack = self.hooks[layer[0][1]][1]
                x = Concat().to(self.device)([x, featureBack])
                self.hooks.append([self.index, x])
                self.index += 1
                pointer += 1
        return self.hooks

class YoloV5m(nn.Module):
    def __init__(self, in_channels, selectLayer ,features, anchor_boxes, device):
        super(YoloV5m, self).__init__()
        self.features = features
        self.anchor_boxes = anchor_boxes
        self.device = device
        self.selectLayer = selectLayer
        
        self.backbone = YoloV5_backbone(in_channels=in_channels,device= self.device)
        self.head = YoloV5_head(1024, self.device)
        
        self.layerP3 = CBSBlock(in_channels=256, 
                              out_channels=self.features, 
                              kernel_size = 1, 
                              padding = 0, 
                              groups = self.features).to(device=self.device)
        self.layerP4 = CBSBlock(in_channels=512, 
                              out_channels=self.features, 
                              kernel_size = 1, 
                              padding = 0, 
                              groups = self.features).to(device=self.device)
        self.layerP5 = CBSBlock(in_channels=1024, 
                              out_channels=self.features, 
                              kernel_size = 1, 
                              padding = 0,
                              groups = self.features).to(device=self.device)
    # input(3x640x640)
    def forward(self, x):
        hooks = self.backbone(x)
        self.head.hooks = [*hooks]
        
        self.head.index = len(hooks)
        hooks = self.head(hooks[len(hooks) - 1][1])
        # chứa 3 khối đầu ra nguyên thủy gồm:
        # + 1024x20x20
        # + 512x40x40
        # + 256x80x80
        layerResuilt = [hooks[i][1] for i in self.selectLayer]
        # transformed = [
        #     self.layerP3(layerResuilt[0]).permute(0, 2, 3, 1)[..., 1:].reshape(-1, 80, 80, 3, 5),
        #     self.layerP4(layerResuilt[1]).permute(0, 2, 3, 1)[..., 1:].reshape(-1, 40, 40, 3, 5),
        #     self.layerP5(layerResuilt[2]).permute(0, 2, 3, 1)[..., 1:].reshape(-1, 20, 20, 3, 5)
        # ]
        return layerResuilt
    
device = "cuda"
model = YoloV5m(in_channels=3, selectLayer=[17, 20, 23] , features=16, anchor_boxes=get_anchor_boxes(), device=device).to(device)
model.eval()
a = torch.randn(17, 3, 640, 640).to(device)

import time
start = time.time()
with torch.no_grad():
    x = model(a)
end = time.time()
print((end - start), " seconds")
print(x[0].shape, x[1].shape, x[2].shape)