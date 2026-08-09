from collections.abc import Sequence
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import torch
from torch import Tensor

def bbox_to_rect(bbox, color):
    """
    bbox: (xmin, ymin, xmax, ymax) 형태의 numpy array 또는 list
    color: 박스 색상 (문자열)
    """
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin

    return patches.Rectangle(
        (xmin, ymin),   # 왼쪽 위 좌표
        width,          # 너비
        height,         # 높이
        fill=False,
        edgecolor=color,
        linewidth=2
    )
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def multibox_prior(data: Tensor, sizes, ratios):
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = num_sizes + num_ratios - 1
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    
    offset_h, offset_w = 0.5, 0.5
    
    #for normalize
    steps_h = 1.0/in_height
    steps_w = 1.0/in_width
    
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing="ij")
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
    
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), size_tensor[0] * torch.sqrt(ratio_tensor[1:]))) * (in_height / in_width)
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), size_tensor[0] / torch.sqrt(ratio_tensor[1:])))
    
    anchor_offsets = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    grid_centers = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
    grid_centers = grid_centers.repeat_interleave(boxes_per_pixel, dim=0)
    output = grid_centers + anchor_offsets
    return output.unsqueeze(0)

def show_bboxes(axes: plt.Axes, bboxes: Tensor, labels: Sequence[str] | None = None, colors: Sequence[str] | None = None):
    def make_list(obj, default_values=None):
        if obj is None: obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ["b", "g", "r", "m", "c"])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center', fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))

def box_iou(boxes1:Tensor, boxes2:Tensor):
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
    
    #return type : if there's 2 gt and 5 anchors
    #               GT 0    GT 1
    # anchor 0      0.20    0.10
    # anchor 1      0.70    0.15
    # anchor 2      0.30    0.60
    # anchor 3      0.10    0.40
    # anchor 4      0.20    0.80

#ground_truth = [class, xmin, ymin, xmax, ymax]
def assign_anchor_to_bbox(ground_truth:Tensor, anchors:Tensor, device, iou_threshold=0.5):
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)
    anchors_bbox_map = torch.full([num_anchors], -1, dtype=torch.long, device=device)
    
    max_ious, indices = torch.max(jaccard, dim=1)
    #max_ious : [0.20, 0.70, 0.60, 0.40, 0.80] <- selected max iou for each anchor
    #indices : [0,0,1,1,1] <- index for correspond gt
    
    
    
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1) #it returns index
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = max_idx % num_gt_boxes
        anc_idx = max_idx // num_gt_boxes
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = -1
        jaccard[anc_idx, :] = -1
    return anchors_bbox_map

def offset_boxes(anchors:Tensor, assigned_bb:Tensor, eps=1e-6):
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = (
        10
        * (c_assigned_bb[:, :2] - c_anc[:, :2])
        / c_anc[:, 2:]
    )

    offset_wh = 5 * torch.log(
        eps + c_assigned_bb[:, 2:] / c_anc[:, 2:]
    )
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset

def multibox_target(anchors, labels) -> tuple[Tensor, Tensor, Tensor]:
    """
    anchors: (a, 4)
    labels: (b,g,5)
    """
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [],[],[]
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        #label = tensor([
        #[2, 0.1, 0.2, 0.4, 0.6],  # GT 0: class 2
        #[0, 0.5, 0.3, 0.9, 0.8],  # GT 1: class 0
        #])
        label = labels[i, :, :] # [class_id, xmin, ymin, xmax, ymax]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1,4)
        
        #init
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true] # positive anchor들이 배정받은 GT 번호
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:] # only bb 4
        
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

def display_anchors(fmap_w, fmap_h, s, image:Tensor, axes:plt.Axes):
    plt.figure(figsize=(3.5, 2.5))
    w,h = image.shape[1:]
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    anchors = multibox_prior(fmap, sizes = s, ratios = [1,2,0.5])
    bbox_scale = torch.tensor([w,h,w,h])
    show_bboxes(axes, anchors[0] * bbox_scale)
    
