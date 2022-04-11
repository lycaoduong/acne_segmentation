from collections import namedtuple, OrderedDict
import yaml
import torch
from torch import nn
import math
import numpy as np
from torchvision.ops.boxes import batched_nms
import os
import cv2
import uuid
import webcolors
from typing import Union
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def create_namedtuple_from_dict(obj):
    if isinstance(obj, dict):
        fields = sorted(obj.keys())
        namedtuple_type = namedtuple(
            typename='GenericObject',
            field_names=fields,
            rename=True,
        )
        field_value_pairs = OrderedDict(
            (str(field), create_namedtuple_from_dict(obj[field]))
            for field in fields
        )
        try:
            return namedtuple_type(**field_value_pairs)
        except TypeError:
            # Cannot create namedtuple instance so fallback to dict (invalid attribute names)
            return dict(**field_value_pairs)
    elif isinstance(obj, (list, set, tuple, frozenset)):
        return [create_namedtuple_from_dict(item) for item in obj]
    else:
        return obj

class parameters_read:
    def __init__(self, params_path):
        self.params = yaml.safe_load(open(params_path, encoding='utf-8').read())

    def update(self, dictionary):
        self.params = dictionary

    def __getattr__(self, item):
        return self.params.get(item, None)

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU

class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        dim = len(anchors.shape) - 1

        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=dim)

def ciou(bboxes1, bboxes2):
    bboxes1 = torch.sigmoid(bboxes1)
    bboxes2 = torch.sigmoid(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True
    w1 = torch.exp(bboxes1[:, 2])
    h1 = torch.exp(bboxes1[:, 3])
    w2 = torch.exp(bboxes2[:, 2])
    h2 = torch.exp(bboxes2[:, 3])
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = bboxes1[:, 0]
    center_y1 = bboxes1[:, 1]
    center_x2 = bboxes2[:, 0]
    center_y2 = bboxes2[:, 1]

    inter_l = torch.max(center_x1 - w1 / 2,center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2,center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2,center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2,center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l), min=0) * torch.clamp((inter_b - inter_t), min=0)

    c_l = torch.min(center_x1 - w1 / 2,center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2,center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2,center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2,center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = torch.clamp((c_r - c_l), min=0)**2 + torch.clamp((c_b - c_t), min=0)**2

    union = area1+area2-inter_area
    u = (inter_diag) / c_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
    with torch.no_grad():
        S = (iou>0.5).float()
        alpha = S*v/(1-iou+v)

    cious = iou - u - alpha * v
    cious = torch.clamp(cious, min=-1.0, max=1.0)
    if exchange:
        cious = cious.T
    return torch.mean(1 - cious)
    # return torch.sum(1 - cious)

def diou(bboxes1, bboxes2):
    bboxes1 = torch.sigmoid(bboxes1)
    bboxes2 = torch.sigmoid(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = torch.zeros((cols, rows))
        exchange = True
    w1 = torch.exp(bboxes1[:, 2])
    h1 = torch.exp(bboxes1[:, 3])
    w2 = torch.exp(bboxes2[:, 2])
    h2 = torch.exp(bboxes2[:, 3])
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = bboxes1[:, 0]
    center_y1 = bboxes1[:, 1]
    center_x2 = bboxes2[:, 0]
    center_y2 = bboxes2[:, 1]

    inter_l = torch.max(center_x1 - w1 / 2,center_x2 - w2 / 2)
    inter_r = torch.min(center_x1 + w1 / 2,center_x2 + w2 / 2)
    inter_t = torch.max(center_y1 - h1 / 2,center_y2 - h2 / 2)
    inter_b = torch.min(center_y1 + h1 / 2,center_y2 + h2 / 2)
    inter_area = torch.clamp((inter_r - inter_l),min=0) * torch.clamp((inter_b - inter_t),min=0)

    c_l = torch.min(center_x1 - w1 / 2,center_x2 - w2 / 2)
    c_r = torch.max(center_x1 + w1 / 2,center_x2 + w2 / 2)
    c_t = torch.min(center_y1 - h1 / 2,center_y2 - h2 / 2)
    c_b = torch.max(center_y1 + h1 / 2,center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    c_diag = torch.clamp((c_r - c_l),min=0)**2 + torch.clamp((c_b - c_t),min=0)**2

    union = area1+area2-inter_area
    u = (inter_diag) / c_diag
    iou = inter_area / union
    dious = iou - u
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T
    # return torch.sum(1 - dious)
    return torch.mean(1 - dious)


def giou(gt_bboxes, pr_bboxes, reduction='mean'):
    """
    gt_bboxes: tensor (-1, 4) xyxy
    pr_bboxes: tensor (-1, 4) xyxy
    loss proposed in the paper of giou
    """
    gt_area = (gt_bboxes[:, 2]-gt_bboxes[:, 0])*(gt_bboxes[:, 3]-gt_bboxes[:, 1])
    pr_area = (pr_bboxes[:, 2]-pr_bboxes[:, 0])*(pr_bboxes[:, 3]-pr_bboxes[:, 1])

    # iou
    lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = gt_area + pr_area - inter
    iou = inter / union
    # enclosure
    lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    enclosure = wh[:, 0] * wh[:, 1]

    giou = iou - (enclosure-union)/enclosure
    loss = 1. - giou
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    return loss

class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes

def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes,
                score_threshold, nms_threshold, nms_method='hard_nms',
                co_coef=0, pyramid_sup='p7'):

    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)

    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > score_threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        if co_coef == 0:
            p3_idx = 36864
            p4_idx = 9216
            p5_idx = 2304
            p6_idx = 576
        elif co_coef == 1:
            p3_idx = 57600
            p4_idx = 14400
            p5_idx = 3600
            p6_idx = 900
        elif co_coef == 2:
            p3_idx = 82944
            p4_idx = 20736
            p5_idx = 5184
            p6_idx = 1296
        elif co_coef == 3:
            p3_idx = 112896
            p4_idx = 28224
            p5_idx = 7056
            p6_idx = 1764

        if pyramid_sup == 'p7':
            sup_list = p3_idx + p4_idx + p5_idx + p6_idx
        elif pyramid_sup == 'p6_7':
            sup_list = p3_idx + p4_idx + p5_idx
        elif pyramid_sup == 'p5_6_7':
            sup_list = p3_idx + p4_idx

        sup_anchor = transformed_anchors[i][:sup_list]
        sup_class = classification[i][:sup_list]
        sup_score = scores[i][:sup_list]
        sup_score_tsh = scores_over_thresh[i][:sup_list]

        classification_per = torch.unsqueeze(sup_class, 0)[i, torch.unsqueeze(sup_score_tsh, 0)[i, :], ...].permute(1, 0)

        # transformed_anchors_per = torch.unsqueeze(sup_anchor, 0)[i,  torch.unsqueeze(sup_score_tsh, 0)[i, :], ...]
        # scores_per = torch.unsqueeze(sup_score, 0)[i, torch.unsqueeze(sup_score_tsh, 0)[i, :], ...]

        transformed_anchors_per = torch.unsqueeze(sup_anchor, 0)[i,  torch.unsqueeze(sup_score_tsh, 0)[i, :], ...].cuda()
        scores_per = torch.unsqueeze(sup_score, 0)[i, torch.unsqueeze(sup_score_tsh, 0)[i, :], ...].cuda()

        if len(classification_per[i]) < 1:
            return 'no_result'

        scores_, classes_ = classification_per.max(dim=0)

        if nms_method == 'hard_nms':
            anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0],
                                                 classes_, nms_threshold)


        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                # 'rois': boxes_.cpu().numpy(),
                # 'class_ids': classes_.cpu().numpy(),
                # 'scores': scores_.cpu().numpy(),
                'rois': boxes_.cpu().detach().numpy(),
                'class_ids': classes_.cpu().detach().numpy(),
                'scores': scores_.cpu().detach().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out

def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])

def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index

def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name) - 36):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard

def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue, rgb_color.green, rgb_color.red)
    return result


STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua', 'Beige', 'Azure', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

color_list = standard_to_bgr(STANDARD_COLORS)

def resize_img(image, img_size):
    height, width, _ = image.shape
    if height > width:
        scale = img_size / height
        resized_height = img_size
        resized_width = int(width * scale)
    else:
        scale = img_size / width
        resized_height = int(height * scale)
        resized_width = img_size

    image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    new_image = np.zeros((img_size, img_size, 3))
    new_image[0:resized_height, 0:resized_width] = image
    return {'img': torch.from_numpy(new_image).to(torch.float32), 'scale': scale}

def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds

def confusion_matrix_plot(true, predict, normalize = False):
    true = true.flatten()
    predict = predict.flatten()
    if normalize:
        conf_mat = confusion_matrix(true, predict, normalize='true')
    else:
        conf_mat = confusion_matrix(true, predict, normalize='false')
    return conf_mat

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):

    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)