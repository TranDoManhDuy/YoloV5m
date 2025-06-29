def get_nc():
    return 1
def get_depth_multiple():
    return 0.67
def get_width_multiple():
    return 0.75
# Pre-calculated Anchor boxes from phase1_Kmean
# 20, 40, 80 => Xem lại phân anchor nào vào khối dự đoán nào
def get_anchor_boxes():
    return [
        [[20.387419637638807, 52.0374780829924], 
        [25.620321332936626, 103.58857482891996], 
        [37.816198391905324, 141.468515674406]], 
        [[75.10403309203723, 196.98376421923476], 
        [84.68336483931947, 388.33756413718606], 
        [169.98925537231383, 262.015992003998]], 
        [[156.22530027121272, 478.2051530414568], 
        [280.7831671779141, 489.80540644171776], 
        [506.06684315023165, 567.2127729980145]]
    ]
# Architecture
# YOLOv5 v6.0 backbone
def get_backbone_config():
    return [
        # Ở các lớp conv, thứ tự các tham số lần lượt là out_channels, kernel_size, stride, padding
        # [from, number, module, args]
        [-1, 1, "conv", [64, 6, 2, 2]], # 0 - P1/2 => 320 * 320 * 64
        [-1, 1, "conv", [128, 3, 2, 1]],   # 1 - P2/4 => Giảm còn 1/4 => 160 * 160 * 128
        [-1, 3, "c3", [128]],
        [-1, 1, "conv", [256, 3, 2, 1]],   # 3 - P3/8 ==> 256x80x80
        [-1, 6, "c3", [256]],
        [-1, 1, "conv", [512, 3, 2, 1]],    # 5 - P4/16 512x40x40
        [-1, 9, "c3", [512]],
        [-1, 1, "conv", [1024, 3, 2, 1]],  # 7 - P5/32 => => 20 * 20 * 1024
        [-1, 3, "c3", [1024]], # => 20 * 20 * 1024
        [-1, 1, "sppf", [1024, 5]] #9 => 20 * 20 * 1024
    ]
def get_head_config():
    return [
        # input khoi HEAD 1024x20x20
        [-1, 1, "conv", [512, 1, 1, 0]], # 10 --> 512x20x20
        [-1, 1, "upsample", [None, 2, "nearest"]], # --> 512x40x40
        [[-1, 6], 1, "concat", [1]], # cat backbone P4 -> # 512x40x40 CONCAT 512x40x40 => 1024x40x40
        [-1, 3, "c3", [512, False]], # 13 ==> 512x40x40
        [-1, 1, "conv", [256, 1, 1, 0]], # ==> 256x40x40
        [-1, 1, "upsample", [None, 2, "nearest"]], # ==> 256x80x80
        [[-1, 4], 1, "concat", [1]], # cat backbone P3  ==>  512x80x80
        [-1, 3, "c3", [256, False]], # 17 (P3/8 - small) => 256x80x80
        [-1, 1, "conv", [256, 3, 2, 1]],           # ==> 256x40x40
        [[-1, 14], 1, "concat", [1]],           # ==> 512x40x40
        [-1, 3, "c3", [512, False]], # 20 (P4/16-medium)    ==> 512x40x40
        [-1, 1, "conv", [512, 3, 2, 1]], #                     ==> 512x20x20
        [[-1, 10], 1, "concat", [1]], # cat head P5         ==> 1024x20x20
        [-1, 3, "c3", [1024, False]], # 23 (P5/32-large)    ==> 1024x20x20
        # [[17, 20, 23], 1, "detect", [get_nc(), get_anchor_boxes()]], # Detect(P3, P4, P5)
    ]