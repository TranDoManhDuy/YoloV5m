import torch
import torch.nn as nn

nc = 80
depth_multiple = 0.67
width_multiple = 0.75 

# Pre-calculated Anchor boxes from phase1_Kmean
def get_anchor_boxes():
    return [
        [20.387419637638807, 52.0374780829924], 
        [25.620321332936626, 103.58857482891996], 
        [37.816198391905324, 141.468515674406], 
        [75.10403309203723, 196.98376421923476], 
        [84.68336483931947, 388.33756413718606], 
        [169.98925537231383, 262.015992003998], 
        [156.22530027121272, 478.2051530414568], 
        [280.7831671779141, 489.80540644171776], 
        [506.06684315023165, 567.2127729980145]
    ]
# Architecture

# YOLOv5 v6.0 backbone
def get_backbone_config():
    return [
        # [from, number, module, args]
        [-1, 1, "conv", [64, 6, 2, 2]], # 0 - P1/2
        [-1, 1, "conv", [128, 3, 2]],   # 1 - P2/4
        [-1, 3, "c3", [128]],
        [-1, 1, "conv", [256, 3, 2]],   # 3 - P3/8
        [-1, 6, "c3", [256]],
        [-1, 1, "conv", [512, 3, 2]],    # 5 - P4/16
        [-1, 9, "c3", [512]],
        [-1, 1, "conv", [1024, 3, 2]],  # 7 - P5/32
        [-1, 3, "c3", [1024]],
        [-1, 1, "sppf", [1024, 5]] #9
    ]

def get_head_config():
    return [
        [-1, 1, "conv", [512, 1, 1]],
        [-1, 1, "upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "concat", [1]], # cat backbone P4
        [-1, 3, "c3", [512, False]], # 13
        [-1, 1, "conv", [256, 1, 1]],
        [-1, 1, "upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "concat", [1]], # cat backbone P3
        [-1, 3, "c3", [256, False]], # 17 (P3/8 - small)
        [-1, 1, "conv", [256, 3, 2]],
        [[-1, 14], 1, "concat", [1]],
        [-1, 3, "c3", [512, False]], # 20 (P4/16-medium)
        [-1, 1, "conv", [512, 3, 2]],
        [[-1, 10], 1, "concat", [1]], # cat head P5
        [-1, 3, "c3", [1024, False]], # 23 (P5/32-large)
        [[17, 20, 23], 1, "detect", [nc, get_anchor_boxes()]], # Detect(P3, P4, P5)
    ]