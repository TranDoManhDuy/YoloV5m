import torch
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8
BATCH_SIZE = 32
IMAGE_SIZE = 640
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMAGE_TRAIN_PATH = "D:\\WorkSpace\\YOLO_You Only Look Once\\SourceYoloV5m\\train\\images"
LABLE_TRAIN_PATH = "D:\\WorkSpace\\YOLO_You Only Look Once\\SourceYoloV5m\\train\\labels"
CSV_FILE_TRAIN = "D:\\WorkSpace\\YOLO_You Only Look Once\\SourceYoloV5m\\data_train_phase0.csv"

scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_REFLECT_101,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),

        A.ColorJitter(
            brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4
        ),

        A.OneOf(
            [
                A.ElasticTransform(
                    alpha=50,
                    sigma=5,
                    p=0.9,
                    border_mode=cv2.BORDER_REFLECT_101
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.5,
                    p=0.9,
                    border_mode=cv2.BORDER_REFLECT_101
                ),
            ],
            p=0.8,
        ),

        A.OneOf(
            [
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Transpose(p=1.0),
            ],
            p=0.9,
        ),

        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

# TEST train_transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
img_path = "D:\\WorkSpace\\YOLO_You Only Look Once\\SourceYoloV5m\\train\\images\\000260_jpg.rf.19fffb10eca7302570d8bd38de62bac9.jpg"
label_path = "D:\\WorkSpace\\YOLO_You Only Look Once\\SourceYoloV5m\\train\\labels\\000260_jpg.rf.19fffb10eca7302570d8bd38de62bac9.txt"
bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
image = np.array(Image.open(img_path).convert("RGB"))

augmentations = train_transforms(image=image, bboxes=bboxes)
image = augmentations["image"]
bboxes = augmentations["bboxes"]
# Chuyển ảnh từ Tensor về NumPy để hiển thị
image = image.permute(1, 2, 0).cpu().numpy()  # HWC
image = (image * 255).astype(np.uint8)  # Vì Normalize trước đó
# Vẽ bounding box
H, W, _ = image.shape
print(H, W)
for box in bboxes:
    x_center, y_center, width, height, cls = box
    x1 = int((x_center - width / 2) * W)
    y1 = int((y_center - height / 2) * H)
    x2 = int((x_center + width / 2) * W)
    y2 = int((y_center + height / 2) * H)
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    cv2.putText(image, f"Class {int(cls)}", (x1, y1 - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(255, 0, 0), thickness=1)
# Hiển thị ảnh với bbox
plt.imshow(image)
plt.axis('off')
plt.show()