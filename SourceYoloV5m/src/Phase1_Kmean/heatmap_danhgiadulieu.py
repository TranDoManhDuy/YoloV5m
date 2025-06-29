import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Định nghĩa thư mục chứa file label
file_path = "D:/WorkSpace/YOLO_You Only Look Once/SourceYoloV5m/src/Phase1_Kmean/all_bounding_boxes_phase0_size.csv"
df = pd.read_csv(file_path)

# all_bouding_box
all_bouding_box = df.values.tolist()
all_bouding_box = np.array(all_bouding_box) * 640
all_bouding_box = all_bouding_box.tolist()

# Danh sách tọa độ tâm bounding box
bounding_size = all_bouding_box
x = [i[0] for i in bounding_size]
y = [i[1] for i in bounding_size]
# Vẽ heatmap phân bố bounding boxes
plt.figure(figsize=(8, 6))
sns.kdeplot(x=x, y=y, cmap="Reds", fill=True)
plt.xlabel("X - Center (pixels)")
plt.ylabel("Y - Center (pixels)")
plt.title("Heatmap of Bounding Boxes")
plt.gca().invert_yaxis()  # Lật trục Y cho đúng góc ảnh
plt.savefig("image.jpg")
plt.show()