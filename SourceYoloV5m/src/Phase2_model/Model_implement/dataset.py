from PIL import Image, ImageFile
import pandas as pd
import torch
import numpy as np
import os
import config
import architecture 
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from utils import *

class YoloV5mDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size=config.IMAGE_SIZE, S=[20, 40, 80], C=1, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.C = C
        self.anchors = torch.tensor(anchors)
        self.ignore_iou_thresh = 0.5
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        lable_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        print(lable_path + "*********************")
        bboxes = np.roll(np.loadtxt(fname=lable_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # ta co 3 khoi
        target = [torch.zeros(s, s, self.num_anchors // 3, 5) for s in self.S] #  x, y, w, h, objectness score
        for box in bboxes:
            print(box, self.anchors.shape)
            # ious_anchors = bbox_ciou_vectorized(torch.tensor(box[0:4]), self.anchors)
        return image, bboxes
if __name__ == "__main__":
    dataset = YoloV5mDataset(
        csv_file=config.CSV_FILE_TRAIN, 
        img_dir=config.IMAGE_TRAIN_PATH, 
        label_dir=config.LABLE_TRAIN_PATH, 
        anchors=architecture.get_anchor_boxes(), 
        transform=config.train_transforms)
    a, b = dataset.__getitem__(0)
    
    plt.imshow(np.transpose(a, (1, 2, 0)))
    plt.axis('off')  # Ẩn trục tọa độ
    plt.show()