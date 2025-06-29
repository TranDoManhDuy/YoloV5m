import os
import csv
import pandas as pd
import numpy as np 
import copy
import matplotlib.pyplot as plt

# function tính IoU khi 2 block cùng tâm nhưng khác kích thước width height
def conpute_iou(box1, box2):
    intersection = min(box1[0], box2[0]) * min(box1[1], box2[1])
    # tổng 2 diện tích trừ đi giao
    union = box1[0] * box1[1] + box2[0] * box2[1] - intersection

    return intersection / (union)

# Đọc file CSV
file_path = "D:/WorkSpace/YOLO_You Only Look Once/SourceYoloV5s/src/Phase1_Kmean/all_bounding_boxes_phase0_size.csv"
df = pd.read_csv(file_path)

# all_bouding_box
all_bouding_box = df.values.tolist()
all_bouding_box = np.array(all_bouding_box) * 640
all_bouding_box = all_bouding_box.tolist()
# chọn điểm khởi tạo ban đầu của thuật toán K-Mean++ chính là điểm có mật độ cao nhất dựa trên mật độ bản đồ kích thước.

# khởi tạo thêm 8 điểm Kmean start
start_k_mean = [35, 135]

# Danh sách tâm cụm đã chọn (giả sử đã có một cụm đầu tiên)
centroids = [start_k_mean]

K = 8 # tìm thêm 8 kích thước phân cụm khác nữa.
for i in range(K):
    # khoảng cách bé nhất từ mỗi bounding box đến các phân cụm gần đã tìm thấy. 
    iou_bounding_box_min_toCentrods = []
    
    # lặp qua từng box và tính khoảng cách của box đó tới các phân cụm.
    for bounding_box in all_bouding_box:
        centroids_temp = copy.deepcopy(centroids)
        
        # tính khoảng cách của bounding box với từng phân cụm
        D = [1 - conpute_iou(bounding_box, d) for d in centroids_temp]
        
        iou_bounding_box_min_toCentrods.append(min(D))

    # iou_bounding_box_min_toCentrods chứa khoảng cách nhỏ nhất của từng điểm đến các centroids.
    D_square = np.array(iou_bounding_box_min_toCentrods)
    D_square = D_square ** 2
    D_square_sum = np.sum(D_square)    
    D_square_probability =  D_square / D_square_sum
    
    # Vị trí max
    index = np.argmax(D_square_probability) 
     
    centroids.append(all_bouding_box[index])
    
file_path_k_mean_start = "D:/WorkSpace/YOLO_You Only Look Once/SourceYoloV5s/src/Phase1_Kmean/generatorKmean.csv"
tf = pd.read_csv(file_path_k_mean_start)
centroids = tf.values.tolist()
# csv_k_mean_generator = "src/Phase1_Kmean/generatorKmean.csv"
# # Ghi dữ liệu vào file CSV
# with open(csv_k_mean_generator, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     # Ghi dòng tiêu đề
#     writer.writerow(["w_start", "y_start"])
#     # Ghi từng dòng dữ liệu
#     writer.writerows(centroids)

# thuật toán K-Mean với 9 
rs = []
x_index = []

epoches = 150
for indexing in range(epoches):
    # chia phân cụm.
    Clustering = [[], [], [], [], [], [], [], [], []]
    for bounding_box in all_bouding_box:
        D = []
        for cluster in centroids:
            D.append(1 - conpute_iou(bounding_box, cluster))
        D = np.array(D)
        index = np.argmin(D)
        Clustering[index].append(bounding_box)
    sum_of_error = 0
    for indeX_cluster in range(len(Clustering)):
        cluster = Clustering[indeX_cluster]
        leng = len(cluster)
        
        w = np.array([i[0] for i in cluster])
        h = np.array([i[1] for i in cluster])
        newClustor_w = w.sum()/ leng
        newClustor_h = h.sum()/ leng
        
        sum_of_error += conpute_iou(centroids[indeX_cluster],[newClustor_w, newClustor_h]) * 100
        centroids[indeX_cluster] = [newClustor_w, newClustor_h]
    print("Tong sai lenh: ", sum_of_error / 9)
    rs.append(sum_of_error / 9)
    x_index.append(indexing)

# Vẽ biểu đồ
plt.plot(x_index, rs)

# Thêm tiêu đề và nhãn trục
plt.title("Biểu đồ mẫu")
plt.xlabel("Trục kỉ nguyên")
plt.ylabel("Trục tỷ lệ khớp phân cụm")

# Hiển thị biểu đồ
plt.show()

print("Ket qua phan cum KMean")
dem = 0
for i in Clustering:
    print(f"Phân cụm thứ {dem + 1} kich thuoc: {centroids[dem]}:",len(i))
    dem += 1
print(len(centroids))
# đây là kết quả đã chạy thành công trên bộ dữ liệu của tôi.

# csv_k_mean_result = "src/Phase1_Kmean_ditruyen/Kmean_result.csv"
# # Ghi dữ liệu vào file CSV
# with open(csv_k_mean_result, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     # Ghi dòng tiêu đề
#     writer.writerow(["w", "h"])
#     # Ghi từng dòng dữ liệu
#     writer.writerows(centroids)