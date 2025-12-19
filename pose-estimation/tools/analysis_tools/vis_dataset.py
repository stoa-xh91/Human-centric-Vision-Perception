import enum
import json

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from pycocotools import mask as mask_util

coco_file = "/raid/wangxuanhan/datasets/coco2014/annotations/person_keypoints_train2014.json"
im_root = "/raid/wangxuanhan/datasets/coco2014/train2014"
color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]).astype(np.float32)
color_list = (color_list.reshape((-1, 3)) * 255).astype(np.uint8)

def _get_connection_rules():
    KEYPOINT_CONNECTION_RULES = [
        # face
        ("left_ear", "left_eye", (102, 204, 255)),
        ("right_ear", "right_eye", (51, 153, 255)),
        ("left_eye", "nose", (102, 0, 204)),
        ("nose", "right_eye", (51, 102, 255)),
        # upper-body
        ("left_shoulder", "right_shoulder", (255, 128, 0)),
        ("left_shoulder", "left_elbow", (153, 255, 204)),
        ("right_shoulder", "right_elbow", (128, 229, 255)),
        ("left_elbow", "left_wrist", (153, 255, 153)),
        ("right_elbow", "right_wrist", (102, 255, 224)),
        # lower-body
        ("left_hip", "right_hip", (255, 102, 0)),
        ("left_hip", "left_knee", (255, 255, 77)),
        ("right_hip", "right_knee", (153, 255, 204)),
        ("left_knee", "left_ankle", (191, 255, 128)),
        ("right_knee", "right_ankle", (255, 195, 77)),
        # face-upper-body
        ("left_ear", "left_shoulder", (255, 195, 77)),
        ("right_ear", "right_shoulder", (255, 195, 77)),
        # ("nose", "left_shoulder", (255, 195, 77)),
        # ("nose", "right_shoulder", (255, 195, 77)),
        # upper-lower-body
        # ("left_shoulder", "left_hip", (255, 195, 77)),
        # ("right_shoulder", "right_hip", (255, 195, 77)),
    ]
    return KEYPOINT_CONNECTION_RULES

def _get_keypoints_name():
    COCO_PERSON_KEYPOINT_NAMES = (
        "nose",
        "left_eye", "right_eye",
        "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    )
    return COCO_PERSON_KEYPOINT_NAMES

def draw_box(im, bbox, color=(0, 255, 0)):
    im = np.copy(im)
    im = cv2.rectangle(im,(bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
    return im
def draw_limbs(im, kpts, colors=(0, 255, 0)):
    im = np.copy(im)
    kpt_names = _get_keypoints_name()
    name2idx = {}
    for i, name in enumerate(kpt_names):
        name2idx[name] = i
    libms = _get_connection_rules()
    cmap = plt.get_cmap('rainbow')
    tmp_colors = [cmap(i) for i in np.linspace(0, 1, len(libms) + 2)]
    tmp_colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in tmp_colors]

    # limbs = np.asarray(libms).reshape((-1,2))
    mid_shoulder = (kpts[name2idx['right_shoulder'], :2] + kpts[name2idx['left_shoulder'], :2]) / 2.0
    sc_mid_shoulder = np.minimum(kpts[name2idx['right_shoulder'], 2], kpts[name2idx['left_shoulder'], 2])
    mid_hip = (kpts[name2idx['right_hip'], :2] + kpts[name2idx['left_hip'], :2]) / 2.0
    sc_mid_hip = np.minimum(kpts[name2idx['right_hip'], 2], kpts[name2idx['left_hip'], 2])
    nose_idx = name2idx['nose']
    if sc_mid_shoulder > 0 and kpts[2, nose_idx] > 0:
        cv2.line(
            im, (int(mid_shoulder[0]), int(mid_shoulder[1])), tuple(kpts[nose_idx,:2]),
            color=tmp_colors[len(libms)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > 0 and sc_mid_hip > 0:
        cv2.line(
            im, (int(mid_shoulder[0]), int(mid_shoulder[1])), (int(mid_hip[0]), int(mid_hip[1])),
            color=tmp_colors[len(libms) + 1], thickness=2, lineType=cv2.LINE_AA)
    for i, l in enumerate(libms):
        pt_1, pt_2 = kpts[name2idx[l[0]]], kpts[name2idx[l[1]]]
        if kpts[name2idx[l[0]], 2]>0:
            cv2.circle(im, (int(pt_1[0]), int(pt_1[1])), 3, tmp_colors[i])
        if kpts[name2idx[l[0]], 2]>0:
            cv2.circle(im, (int(pt_2[0]), int(pt_2[1])), 3, tmp_colors[i], thickness=-1, lineType=cv2.LINE_AA)
        if kpts[name2idx[l[0]], 2]>0 and kpts[name2idx[l[1]], 2] > 0:
            cv2.line(im, (int(pt_1[0]), int(pt_1[1])), (int(pt_2[0]), int(pt_2[1])), tmp_colors[i], thickness=2, lineType=cv2.LINE_AA)
    return im

def draw_keypoints(img, kpts, col, point_size=1):
    col_len = len(color_list)
    for i, kpt in enumerate(kpts):
        color = color_list[i%col_len]
        cv2.circle(img, (int(kpt[0]), int(kpt[1])), point_size, (int(color[0]), int(color[1]), int(color[2])), 3)
    return img
def draw_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""
    _WHITE = (255, 255, 255)
    img = np.copy(img)
    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)      

def polys_to_mask(polygons, height, width):
        """Convert from the COCO polygon segmentation format to a binary mask
        encoded as a 2D array of data type numpy.float32. The polygon segmentation
        is understood to be enclosed inside a height x width image. The resulting
        mask is therefore of shape (height, width).
        """
        rle = mask_util.frPyObjects(polygons, height, width)
        mask = np.array(mask_util.decode(rle), dtype=np.float32)
        # Flatten in case polygons was a list
        mask = np.sum(mask, axis=2)
        mask = np.array(mask > 0, dtype=np.uint8)
        return mask

def _poly2mask(mask_ann, img_h, img_w):
        """Private function to convert masks represented with polygon to
        bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            numpy.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """

        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = mask_util.frPyObjects(mask_ann, img_h, img_w)
            rle = mask_util.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = mask_util.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = mask_util.decode(rle)
        return mask

def _xywh2cs(x, y, w, h):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    aspect_ratio = 256 * 1.0 / 192
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def _box2cs(box):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h)

def _load_coco_annotation_kernal(self, coco, im_id):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = coco.loadImgs(im_id)[0]
        width = im_ann['width']
        height = im_ann['height']
        file_name = '%012d.jpg' % im_id
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name
        annIds = coco.getAnnIds(imgIds=im_id, iscrowd=False)
        objs = coco.loadAnns(annIds)
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            obj_size = obj['clean_bbox'][2:4]
            rec.append({
                'image': os.path.join(im_root, file_name),
                'center': center,
                'scale': scale,
                'obj_size': obj_size,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec

if __name__ == "__main__":
    
    coco = COCO(coco_file)
    im_ids = coco.getImgIds()
    selected_ids = np.random.choice(im_ids, len(im_ids))
    # selected_ids = [im_ids[randint(0, len(im_ids))]]
    count = 0
    for im_id in selected_ids:
        if count > 2:
            break
        file_name = coco.loadImgs([im_id])[0]['file_name']
        im = cv2.imread(os.path.join(im_root, file_name))
        anns = coco.loadAnns(coco.getAnnIds([im_id]))
        if len(anns)>1:
            continue
        valid = 0
        color_id = 0
        for ann in anns:
            bbr =  np.round(ann['bbox']) 
        # draw keypoints
        image = im.copy()
        height, width = image.shape[:2]
        for ann in anns:
            if ann['num_keypoints']>0:
                valid+=1
            x, y, w, h = ann['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1)))) 
            _box_img = draw_box(image, [int(x1),int(y1),int(x2),int(y2)])
            _box_img = np.copy(_box_img[int(y1-10):int(y2+10), int(x1-10):int(x2+10),...])
            _box_img = cv2.resize(_box_img, (192, 256))
            bpts = np.asarray([[x1+w/2, y1+h/2], [x1+w/2, y1], [x2, y1+h/2], [x1+w/2, y2], [x1, y1+h/2]]).reshape(-1,2)
            for i in range(len(bpts)):
                bpts[i][0] -= x1
                bpts[i][1] -= y1
                bpts[i][0] = int(192 * bpts[i][0] / (x2-x1+1))
                bpts[i][1] = int(256 * bpts[i][1] / (y2-y1+1))

            kpts = np.asarray(ann['keypoints']).reshape((-1,3))
            _kpt_img = draw_limbs(image, kpts)
            _kpt_img = np.copy(_kpt_img[int(y1-10):int(y2+10), int(x1-10):int(x2+10),...])
            _kpt_img = cv2.resize(_kpt_img, (192, 256))
            for i in range(len(kpts)):
                kpts[i][0] -= x1
                kpts[i][1] -= y1
                kpts[i][0] = int(192 * kpts[i][0] / (x2-x1+1))
                kpts[i][1] = int(256 * kpts[i][1] / (y2-y1+1))

            seg_poly = ann['segmentation']
            if len(seg_poly) != 0:
                mask = polys_to_mask(seg_poly, image.shape[0], image.shape[1])
                _mask_img = draw_mask(image, mask, color_list[np.random.randint(len(color_list)) % len(color_list), 0:3])
                _mask_img = np.copy(_mask_img[int(y1-10):int(y2+10), int(x1-10):int(x2+10),...])
                _mask_img = cv2.resize(_mask_img, (192, 256))
                seg_poly = np.asarray(seg_poly[0])
                seg_poly = seg_poly.reshape(-1, 2)
                for i in range(len(seg_poly)):
                    seg_poly[i][0] -= x1
                    seg_poly[i][1] -= y1
                    seg_poly[i][0] = int(192 * seg_poly[i][0] / (x2-x1+1))
                    seg_poly[i][1] = int(256 * seg_poly[i][1] / (y2-y1+1))
            else:
                _mask_img = np.zeros_like(_kpt_img)

            _image = np.zeros((256, 192*3+20, 3), dtype=np.uint8)
            _image[:,:192,] += _box_img
            _image[:,202:394,] += _kpt_img
            _image[:,404:596,] += _mask_img

            cropped_image = np.copy(image[int(y1):int(y2+1), int(x1):int(x2+1),...])
            image = cv2.resize(cropped_image, (192, 256))
            color_id = color_id % len(color_list)
            color = color_list[color_id]
            bpt_img = draw_keypoints(image.copy(), bpts, (int(color[0]), int(color[1]), int(color[2])),3)
            kpt_img = draw_keypoints(image.copy(), kpts, (int(color[0]), int(color[1]), int(color[2])),2)
            seg_img = draw_keypoints(image.copy(), seg_poly, (int(color[0]), int(color[1]), int(color[2])))
            image = np.zeros((256, 192*3+20, 3), dtype=np.uint8)
            image[:,:192,] += bpt_img
            image[:,202:394,] += kpt_img
            image[:,404:596,] += seg_img
            # image = np.concatenate([bpt_img, kpt_img, seg_img], axis=1)
        # draw limbs
        # for ann in anns:
        #     kpts = np.asarray(ann['keypoints']).reshape((-1,3))
            
        #     color_id = color_id % len(color_list)
        #     color = color_list[color_id]
        #     draw_limbs(images[im_id], kpts, (int(color[0]), int(color[1]), int(color[2])))
        #     color_id += 1
        # draw instance mask
        # mask_color_id = 0
        # images = im.copy()
        # for ann in anns:
        #     if max(ann['keypoints']) == 0:
        #         continue
        #     seg_poly = ann['segmentation']
        #     if len(seg_poly) != 0:
        #         mask = polys_to_mask(seg_poly, images[im_id].shape[0], images[im_id].shape[1])
        #     color_mask = color_list[mask_color_id % len(color_list), 0:3]
        #     mask_color_id += 1
        #     images[im_id] = draw_mask(images[im_id], mask, color_mask)
        if valid>0:
            cv2.imwrite("new_style_"+file_name, image)
            cv2.imwrite("old_style_"+file_name, _image)
            count+=1
            break
                    

