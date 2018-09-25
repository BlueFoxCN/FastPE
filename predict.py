import math
from numpy import unravel_index
import matplotlib
from scipy.ndimage.filters import gaussian_filter
import os
from pathlib import Path
import time
import cv2
import pdb
import shutil
import argparse
from tensorpack import *
from operator import itemgetter
from itertools import groupby
import numpy as np


try:
    from .train import Model
    from .reader import Data
    from .cfgs.config import cfg
except Exception:
    from train import Model
    from reader import Data
    from cfgs.config import cfg


def pad_right_down_corner(img, stride, pad_value):
    h, w, _ = img.shape

    pad = 4 * [0]
    pad[2] = 0 if (h % stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + pad_value, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + pad_value, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def initialize(model_path):
    # prepare predictor
    sess_init = SaverRestore(model_path)
    model = Model()
    predict_config = PredictConfig(session_init = sess_init,
                                   model = model,
                                   input_names = ['imgs'],
                                   output_names = ['heatmaps'])#n h w c
    predict_func = OfflinePredictor(predict_config)
    return predict_func    

def detect(img, predict_func):
    h, w, _ = img.shape

    scale_img = cv2.resize(img, (192, 256), interpolation=cv2.INTER_CUBIC)
    
    scale_img_expanded = np.expand_dims(scale_img, axis=0)

    heatmap = predict_func(scale_img_expanded)[0]# n h w c
   
    heatmap_transpose = np.transpose(heatmap, [0, 3, 1, 2])
    num_joints = heatmap_transpose.shape[1]
    # pdb.set_trace()
    heatmaps_reshaped = heatmap_transpose.reshape(1, num_joints,64*48)
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape(1, 17, 1)
    idx = idx.reshape((1, num_joints, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    
    preds[:, :, 0] = (preds[:, :, 0]) % 48
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / 48)


    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
    [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        cv2.circle(img, (int(preds[0][i][0]*(w//48)), int(preds[0][i][1]*(h//64))), 4, colors[i], thickness=-1)

    cv2.imwrite("result.jpg", img)

def predict_imgs(input_imgs, predict_func, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                      [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    total_imgs = os.listdir(input_imgs)
    print("total imgs", len(total_imgs))

    for im_idx, im in enumerate(total_imgs):
        if im_idx >2000:
            break

        image_path = os.path.join(input_imgs, im)
        print(image_path)

        start_time = time.time()

        img = cv2.imread(image_path)
        img = cv2.transpose(img)  
        img = cv2.flip(img,1)
        h, w, _ = img.shape

        scale_img = cv2.resize(img, (192, 256), interpolation=cv2.INTER_CUBIC)
        
        scale_img_expanded = np.expand_dims(scale_img, axis=0)

        heatmap = predict_func(scale_img_expanded)[0]# n h w c
       
        heatmap_transpose = np.transpose(heatmap, [0, 3, 1, 2])
        num_joints = heatmap_transpose.shape[1]
        # pdb.set_trace()
        heatmaps_reshaped = heatmap_transpose.reshape(1, num_joints,64*48)
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape(1, 17, 1)
        idx = idx.reshape((1, num_joints, 1))
        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
        
        preds[:, :, 0] = (preds[:, :, 0]) % 48
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / 48)

        for i in range(17):
            cv2.circle(img, (int(preds[0][i][0]*(w//48)), int(preds[0][i][1]*(h//64))), 4, colors[i], thickness=-1)

        cv2.imwrite(os.path.join(output_dir ,im.split(".")[0]+"_result.jpg"), img)
        print("time ", time.time()- start_time)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model', required = True)
    parser.add_argument('--input_path', help='path of input data')
    parser.add_argument('--input_imgs', help='dirs of input imgs')
    parser.add_argument('--output_dir', help='dirs of input imgs', default='output')
    args = parser.parse_args()

    predict_func = initialize(args.model_path)

    if args.input_imgs != None:#mul-imgs predict
        predict_imgs(args.input_imgs, predict_func, args.output_dir)
    else:
        img = cv2.imread(args.input_path)#single img predict
        detect(img, predict_func)
  