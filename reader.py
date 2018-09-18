from pathlib import Path
import pickle
import os
import sys

sys.path.insert(1, '/home/user/DatasetsLocal/coco/cocoapi/PythonAPI/')
from pycocotools.coco import COCO

from tensorpack import *
import numpy as np
from scipy import misc
import cv2
import json
from itertools import chain
import math
import random
import time
import pdb
import copy
from transforms import get_affine_transform, affine_transform, fliplr_joints

try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def norm(self):
        return np.sqrt(self.x * self.x + self.y * self.y)

class Data(RNGDataFlow):
    def __init__(self, train_or_test, shuffle=True):
        super(Data, self).__init__()

        assert train_or_test in ['train', 'val']
        self.image_set = train_or_test
        self.anno_path = cfg.train_ann if train_or_test == 'train' else cfg.val_ann
        self.labels_dir = cfg.train_labels_dir if train_or_test == 'train' else cfg.val_labels_dir
        # self.masks_dir = cfg.train_masks_dir if train_or_test == 'train' else cfg.val_masks_dir
        # self.images_dir = cfg.train_images_dir if train_or_test == 'train' else cfg.val_images_dir

        self.aspect_ratio = cfg.image_width * 1.0 / cfg.image_height

        self.coco = COCO(self.anno_path)

        # deal with class names
        cats = [cat['name']for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats#['__background__', 'person']
        self.num_classes = len(self.classes)


        # load image file names
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))#{'person': 1, '__background__': 0}
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))#{'person': 1}
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls], self._class_to_ind[cls]) for cls in self.classes[1:]])#{1: 1}

        self.image_set_index = self.coco.getImgIds()
        self.num_images = len(self.image_set_index)#len=118287

        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
      

        
        if train_or_test == "train" or cfg.use_gt_bbox:
            gt_db = []
            for index in self.image_set_index:
                gt_db.extend(self.load_coco_keypoint_annotation_kernal(index))
            self.db = gt_db#len 149813
        else:
            self.db = self.load_coco_person_detection_results()


        if  train_or_test == "train" and cfg.select_data:
            self.db = self.select_data(self.db)#len 145294

        self.shuffle = shuffle
       
        # if cfg.debug == True and cfg.debug_sample_num <= len(self.img_id_list):
        #     self.img_id_list = self.img_id_list[:cfg.debug_sample_num]

    def size(self):
        return len(self.db)

    def load_coco_person_detection_results():
        pass
 

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (cfg.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        # logger.info('=> num db: {}'.format(len(db)))
        # logger.info('=> num selected db: {}'.format(len(db_selected)))

        return db_selected

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / cfg.pixel_std, h * 1.0 / cfg.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25
        return center, scale

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name

        prefix = 'val2017' if self.image_set=="val" else 'train2017'

        # data_name = prefix + '.zip@' if cfg.data_format == 'zip' else prefix

        image_path = os.path.join(
            "coco", prefix, file_name)

        return image_path

    def load_coco_keypoint_annotation_kernal(self, index):
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
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                # obj['clean_bbox'] = [x1, y1, x2, y2]
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

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
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert cfg.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if cfg.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               cfg.heatmap_size[1],
                               cfg.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = cfg.sigma * 3

            for joint_id in range(self.num_joints):
                # feat_stride = [cfg.image_width, cfg.image_height] / cfg.heatmap_size
                feat_stride = [cfg.image_width / cfg.heatmap_size[0] , cfg.image_height / cfg.heatmap_size[1]]
                
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= cfg.heatmap_size[0] or ul[1] >= cfg.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * cfg.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], cfg.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], cfg.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], cfg.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], cfg.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

    def augmentation_scale(self, img, mask, label):
        dice = np.random.rand()
        if dice > self.params["scale_prob"]:
            scale_multiplier = 1
        else:
            dice2 = np.random.rand()
            scale_multiplier = (self.params["scale_max"] - self.params["scale_min"]) * dice2 + self.params["scale_min"]
        scale_abs = self.params["target_dist"] / label["scale_self"]
        scale = scale_abs * scale_multiplier

        img_aug = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        mask_aug = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # modify meta data
        label["objpos"][0] = scale * label["objpos"][0]
        label["objpos"][1] = scale * label["objpos"][1]

        for person in label["persons"]:
            for joint_idx in range(cfg.ch_heats - 1):
                person["joint"][joint_idx, 0] *= scale
                person["joint"][joint_idx, 1] *= scale

        return img_aug, mask_aug, label

    def rotate_point(self, p, R, xy=True):
        if xy == True:
            aug_point = np.asarray(p + [1])
        else:
            aug_point = np.asarray([p[1], p[0], 1])
        r_point = np.matmul(R, aug_point)
        if xy == True:
            return [r_point[0], r_point[1]]
        else:
            return [r_point[1], r_point[0]]

    def augmentation_rotate(self, img, mask, label):

        dice = np.random.rand()
        degree = (dice - 0.5) * 2 * self.params["max_rotate_degree"]
        height, width, _ = img.shape
        center = (width / 2, height / 2)

        # list of vertexes, in (x, y, 1) order
        vertex_list = [
            [0, 0],
            [width-1, 0],
            [0, height-1],
            [width-1, height-1]
        ]

        R = cv2.getRotationMatrix2D(center, degree, 1)
        rotate_vertex_list = []
        for idx in range(4):
            rotate_vertex_list.append(self.rotate_point(vertex_list[idx], R))

        rotate_vertex_ary = np.asarray(rotate_vertex_list)
        bbox_width = np.max(rotate_vertex_ary[:,0]) - np.min(rotate_vertex_ary[:,0])
        bbox_height = np.max(rotate_vertex_ary[:,1]) - np.min(rotate_vertex_ary[:,1])

        R[0, 2] += bbox_width / 2 - center[0]
        R[1, 2] += bbox_height / 2 - center[1]

        img_aug = cv2.warpAffine(img, R, (int(bbox_width), int(bbox_height)), _, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, (128, 128, 128))
        mask_aug = cv2.warpAffine(mask, R, (int(bbox_width), int(bbox_height)), _, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, 255)

        label['objpos'] = self.rotate_point(label['objpos'], R)

        # modify meta data
        label['objpos'] = self.rotate_point(label['objpos'], R, xy=False)

        for person in label["persons"]:
            for joint_idx in range(cfg.ch_heats - 1):
                joint = [person['joint'][joint_idx, 0], person['joint'][joint_idx, 1]]
                rotate_joint = self.rotate_point(joint, R)
                person["joint"][joint_idx, 0] = rotate_joint[0]
                person["joint"][joint_idx, 1] = rotate_joint[1]

        return img_aug, mask_aug, label

    def augmentation_crop(self, img, mask, label):
        dice_x = np.random.rand()
        dice_y = np.random.rand()

        x_offset = (dice_x - 0.5) * 2 * self.params["center_perterb_max"]
        y_offset = (dice_y - 0.5) * 2 * self.params["center_perterb_max"]

        height, width, _ = img.shape

        center_x = min(max(label["objpos"][0] + x_offset, 0), width - 1)
        center_y = min(max(label["objpos"][1] + y_offset, 0), height - 1)

        x_offset = center_x - label["objpos"][0]
        y_offset = center_y - label["objpos"][1]

        start_x = int(center_x - self.params["crop_size_x"] * 0.5)
        start_y = int(center_y - self.params["crop_size_y"] * 0.5)
        end_x = start_x + self.params["crop_size_x"]
        end_y = start_y + self.params["crop_size_y"]

        img_aug = np.ones((self.params["crop_size_y"], self.params["crop_size_x"], 3)) * 128
        mask_aug = np.ones((self.params["crop_size_y"], self.params["crop_size_x"])) * 255

        crop_start_x = max(start_x, 0)
        crop_end_x = min(end_x, width)
        crop_start_y = max(start_y, 0)
        crop_end_y = min(end_y, height)
        crop_width = crop_end_x - crop_start_x
        crop_height = crop_end_y - crop_start_y

        offset_x = 0 if start_x >= 0 else -start_x
        offset_y = 0 if start_y >= 0 else -start_y

        img_aug[offset_y:offset_y+crop_height, offset_x:offset_x+crop_width] = img[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
        mask_aug[offset_y:offset_y+crop_height, offset_x:offset_x+crop_width] = mask[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

        # modify meta data
        offset_left = self.params["crop_size_x"] / 2 - label["objpos"][0] - x_offset
        offset_up = self.params["crop_size_y"] / 2 - label["objpos"][1] - y_offset

        label["objpos"] = [label["objpos"][0] + offset_left, label["objpos"][1] + offset_up]

        for person in label["persons"]:
            for joint_idx in range(cfg.ch_heats - 1):
                person["joint"][joint_idx, 0] = person["joint"][joint_idx, 0] + offset_left
                person["joint"][joint_idx, 1] = person["joint"][joint_idx, 1] + offset_up
                if person["joint"][joint_idx, 0] <= 0 or person["joint"][joint_idx, 0] >= self.params["crop_size_x"] - 1 \
                   or person["joint"][joint_idx, 1] <= 0 or person["joint"][joint_idx, 1] >= self.params["crop_size_y"] - 1:
                    person["joint"][joint_idx, 2] = 2

        return img_aug, mask_aug, label


    def augmentation_flip(self, img, mask, label):
        dice = np.random.rand()
        do_flip = dice <= self.params["flip_prob"]

        if do_flip == True:
            return img, mask, label
        img_aug = cv2.flip(img, 1)
        mask_aug = cv2.flip(mask, 1)

        label["objpos"][0] = self.params["crop_size_x"] - 1 - label["objpos"][0]

        right_idxes = [2, 3, 4, 8, 9, 10, 14, 16]
        left_idxes =  [5, 6, 7, 11, 12, 13, 15, 17]

        _, w, _ = img_aug.shape

        # need to exchange left joints with right joints
        for person in label["persons"]:
            for joint_idx in range(cfg.ch_heats - 1):
                person["joint"][joint_idx, 0] = w - 1 - person["joint"][joint_idx, 0]
            for idx, joint_idx_1 in enumerate(right_idxes):
                joint_idx_2 = left_idxes[idx]
                # exchange the joint_idx_1-th joint with the joint_idx_2-th joint
                person["joint"][[joint_idx_1, joint_idx_2]] = person["joint"][[joint_idx_2, joint_idx_1]]

        return img_aug, mask_aug, label

    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.db)

        for img_id in range(len(self.db)):
            img_id = np.random.randint(0,len(self.db))
            db_rec = copy.deepcopy(self.db[img_id])
            image_file = db_rec['image']
            filename = db_rec['filename'] if 'filename' in db_rec else ''
            imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''
            
            data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            # data_numpu = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            if data_numpy is None:
                continue

            joints = db_rec['joints_3d']
            joints_vis = db_rec['joints_3d_vis']

            c = db_rec['center']
            s = db_rec['scale']
            score = db_rec['score'] if 'score' in db_rec else 1
            r = 0

            if self.image_set == "train":
                sf = cfg.scale_factor
                rf = cfg.rot_factor
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                    if random.random() <= 0.6 else 0

                if cfg.flip and random.random() <= 0.5:
                    data_numpy = data_numpy[:, ::-1, :]
                    joints, joints_vis = fliplr_joints(
                        joints, joints_vis, data_numpy.shape[1], [])
                    c[0] = data_numpy.shape[1] - c[0] - 1

            trans = get_affine_transform(c, s, r, [cfg.image_width, cfg.image_height])
            input = cv2.warpAffine(data_numpy, trans, (int(cfg.image_width), int(cfg.image_height)), flags=cv2.INTER_LINEAR)
            input1=copy.deepcopy(input)
               

            if cfg.use_normalize == True:
                input = input / 255
                mean = [0.485, 0.456, 0.406]    # rgb
                std = [0.229, 0.224, 0.225]
                if cfg.is_rgb == True:
                    input = (input-mean) / std
                else:
                    input = (input-mean[::-1]) / std[::-1]

            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

            target, target_weight = self.generate_target(joints, joints_vis)

            # target = torch.from_numpy(target)
            # target_weight = torch.from_numpy(target_weight)

            meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score}
            # pdb.set_trace()
            #target (17, 64, 64), target_weight (17, 1),
            # yield input, target, target_weight, meta
            yield input, target, target_weight

if __name__ == '__main__':
    ds = Data('train', True)
    ds.reset_state()
    g = ds.get_data()
    sample = next(g)
