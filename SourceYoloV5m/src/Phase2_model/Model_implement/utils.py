import torch
import math

# complete IOU calculation
def bbox_ciou_vectorized(box1, box2, eps=1e-7):
    """
    Compute CIoU between two sets of boxes.
    box1: (batch_size, N, 4) in (cx, cy, w, h)
    box2: (batch_size, M, 4) in (cx, cy, w, h)
    Return: (batch_size, N, M) CIoU matrix
    """
    # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
    b1_x1 = box1[..., 0] - box1[..., 2] / 2
    b1_y1 = box1[..., 1] - box1[..., 3] / 2
    b1_x2 = box1[..., 0] + box1[..., 2] / 2
    b1_y2 = box1[..., 1] + box1[..., 3] / 2

    b2_x1 = box2[..., 0] - box2[..., 2] / 2
    b2_y1 = box2[..., 1] - box2[..., 3] / 2
    b2_x2 = box2[..., 0] + box2[..., 2] / 2
    b2_y2 = box2[..., 1] + box2[..., 3] / 2

    # Intersection
    inter_x1 = torch.max(b1_x1[..., None], b2_x1)
    inter_y1 = torch.max(b1_y1[..., None], b2_y1)
    inter_x2 = torch.min(b1_x2[..., None], b2_x2)
    inter_y2 = torch.min(b1_y2[..., None], b2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1[..., None] + area2 - inter_area + eps
    iou = inter_area / union
    
    # center distance squared
    center_dist = (box1[..., 0][..., None] - box2[..., 0]) ** 2 + (box1[..., 1][..., None] - box2[..., 1]) ** 2

    # enclosing box diagonal squared
    c_x1 = torch.min(b1_x1[..., None], b2_x1)
    c_y1 = torch.min(b1_y1[..., None], b2_y1)
    c_x2 = torch.max(b1_x2[..., None], b2_x2)
    c_y2 = torch.max(b1_y2[..., None], b2_y2)
    c_diag = ((c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2) + eps
    
    # aspect ratio penalty
    atan1 = torch.atan(box1[..., 2][..., None] / (box1[..., 3][..., None] + eps))
    atan2 = torch.atan(box2[..., 2] / (box2[..., 3] + eps))
    v = (4 / math.pi ** 2) * (atan2 - atan1) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - center_dist / c_diag - alpha * v
    return ciou

# input: bboxes = [x1, y1, x2, y2, score]
def non_max_suppression(bboxes, threshold=0.5, iou_threshold=0.4):
    bboxes = [box for box in bboxes if box[4] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if bbox_ciou_vectorized(
                chosen_box[:4],
                box[:4]
            )
            < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms
def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    """
    # predictions: (batch_size, 3, S, S, 5)
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., :4]
    
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 4:5])
    else:
        scores = predictions[..., 4:5]
        
    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )