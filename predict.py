import math
from numpy import unravel_index
import matplotlib
from scipy.ndimage.filters import gaussian_filter
import os
from pathlib import Path
import time
import cv2
import pdb

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
                                   output_names = ['heatmaps'])
    predict_func = OfflinePredictor(predict_config)
    return predict_func

'''
def fast_detect(img, predict_func, scale=1, draw_result=False):
    print("0: %f" % time.time())
    h, w, _ = img.shape

    scale_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else img
    scale_img_expanded = np.expand_dims(scale_img, axis=0)

    print("1: %f" % time.time())
    heatmap, paf = predict_func(scale_img_expanded)
    print("2: %f" % time.time())

    heatmap = cv2.resize(heatmap[0], (0,0), fx=cfg.stride, fy=cfg.stride, interpolation=cv2.INTER_CUBIC)

    peaks = []
    print("3: %f" % time.time())

    for part in range(cfg.ch_heats - 1):
        part_heatmap = heatmap[:, :, part]
        y, x = unravel_index(part_heatmap.argmax(), part_heatmap.shape)
        # for opencv to draw, the x is before y
        peaks.append((x, y, part_heatmap[y, x]))
    print("4: %f" % time.time())

    canvas = np.copy(scale_img) # B,G,R order

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    
    cmap = matplotlib.cm.get_cmap('hsv')
    for i in range(cfg.ch_heats - 1):
        rgba = np.array(cmap(1 - i/18. - 1./36))
        rgba[0:3] *= 255
        cv2.circle(canvas, peaks[i][0:2], 4, colors[i], thickness=-1)
        to_plot = cv2.addWeighted(scale_img, 0.3, canvas, 0.7, 0)
    return to_plot
'''

        

def detect(img, predict_func, draw_result=False):
    h, w, _ = img.shape

    # 1. predict on multi scale images and average the results in different scales
    # multiplier = [x * cfg.img_y / h for x in cfg.scale_search]

  
    scale_img = cv2.resize(img, (192, 256), interpolation=cv2.INTER_CUBIC)
    
    scale_img_expanded = np.expand_dims(scale_img, axis=0)

    heatmap = predict_func(scale_img_expanded)# n h w c
    
    heatmap_transpose = np.transpose(heatmap, [0, 3, 1, 2])
    num_joints = heatmap.shape[1]
    
    heatmaps_reshaped = heatmap_transpose.reshape(1, num_joints,)


        heatmap = cv2.resize(heatmap[0], (0,0), fx=cfg.stride, fy=cfg.stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:scale_img_padded.shape[0] - pad[2], :scale_img_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

      
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        

    raw_heatmap_shown = np.maximum(0, heatmap_avg[:, :, 0:1] * 255)
    heatmap_shown = cv2.applyColorMap(raw_heatmap_shown.astype(np.uint8), cv2.COLORMAP_JET)
    img_with_heatmap = cv2.addWeighted(heatmap_shown, 0.5, img, 0.5, 0)

    if draw_result:
        cv2.imwrite('heatmap_shown.jpg', img_with_heatmap)


    # 2. get the part results
    # each element in all_peaks represents a peak and consists of 4 elements, which are:
    #    1. x-coord
    #    2. y-coord
    #    3. heatmap value
    #    4. peak idx
    all_peaks = []
    peak_counter = 0

    for part in range(cfg.ch_heats - 1):
        map_ori = heatmap_avg[:, :, part]
        map_flt = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map_flt.shape)
        map_left[1:,:] = map_flt[:-1,:]
        map_right = np.zeros(map_flt.shape)
        map_right[:-1,:] = map_flt[1:,:]
        map_up = np.zeros(map_flt.shape)
        map_up[:,1:] = map_flt[:,:-1]
        map_down = np.zeros(map_flt.shape)
        map_down[:,:-1] = map_flt[:,1:]

        peaks_binary = np.logical_and.reduce((map_flt>=map_left, map_flt>=map_right, map_flt>=map_up, map_flt>=map_down, map_flt>cfg.thre1))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks_with_score))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks_with_score)

    # 3. get the connection results
    # each element in connection_all represents a peak and consists of 5 elements, which are:
    #    1. peak indx for start part
    #    2. peak index for end part
    #    3. connection score value
    #    4. peak index for start part within this kind of parts
    #    5. peak index for end part within this kind of parts
    connection_all = []
    # special_k records those pair of parts which do not have connections
    special_k = []
    mid_num = 10

    for k in range(len(cfg.map_idx)):
        # score_mid is the two paf features corresponding to the k-th vector
        score_mid = paf_avg[:, :, cfg.map_idx[k]]
        # cand_a and cand_b are detected peaks for the k-th vector
        cand_a = all_peaks[cfg.limb_seq[k][0]]
        cand_b = all_peaks[cfg.limb_seq[k][1]]
        num_a = len(cand_a)
        num_b = len(cand_b)

        if num_a != 0 and num_b != 0:
            # first choose candidates and calculate their scores
            connection_candidate = []
            for i in range(num_a):
                for j in range(num_b):
                    # vec is the unit vector from cand_a to cand_b
                    vec = np.subtract(cand_b[j][:2], cand_a[i][:2])
                    norm = max(math.sqrt(vec[0]*vec[0] + vec[1]*vec[1]), 1e-5)
                    vec = np.divide(vec, norm)
                    
                    startend = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=mid_num), \
                                        np.linspace(cand_a[i][1], cand_b[j][1], num=mid_num)))
                    
                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    # when the two parts are very close (small norm), it is hard to predict the vector, so the score value in score_midpts may be small
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * img.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > cfg.thre2)[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior + cand_a[i][2] + cand_b[j][2]])

            # from the following code, it seems that only heuristic greedy matching is applied
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if i not in connection[:,3] and j not in connection[:,4]:
                    connection = np.vstack([connection, [cand_a[i][3], cand_b[j][3], s, i, j]])
                    if(len(connection) >= min(num_a, num_b)):
                        break

            connection_all.append(connection)

        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    # flatten all peaks into the candidate list
    candidate = np.array([item for sublist in all_peaks for item in sublist])
    subset = np.ones((0, 20))

    for k in range(len(cfg.map_idx)):
        if k in special_k:
            continue

        # partAs and partBs are global peak indexes for the k-th kind of connections
        partAs = connection_all[k][:,0]
        partBs = connection_all[k][:,1]
        # indexA and indexB are the start and end part index for the k-th kind of connections
        indexA, indexB = np.array(cfg.limb_seq[k])
    
        for i in range(len(connection_all[k])):
            # for each connection in the k-th kind of connections
            found = 0
            subset_idx = [-1, -1]
            # count persons already detected and intersected with the new connection
            for j in range(len(subset)):
                if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                    subset_idx[found] = j
                    found += 1
            
            if found == 1:
                # only one person found, should add this new connection to this person
                j = subset_idx[0]
                if(subset[j][indexB] != partBs[i]):
                    subset[j][indexB] = partBs[i]
                    subset[j][-1] += 1
                    subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
            elif found == 2:
                # two persons found, should merge these two persons
                j1, j2 = subset_idx
                # print("found = 2")
                membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                if len(np.nonzero(membership == 2)[0]) == 0:
                    # the two persons have no conflict, merge them
                    subset[j1][:-2] += (subset[j2][:-2] + 1)
                    subset[j1][-2:] += subset[j2][-2:]
                    subset[j1][-2] += connection_all[k][i][2]
                    subset = np.delete(subset, j2, 0)
                else: # as like found == 1
                    # The person in the former has more confidence and should be updated with higher priority
                    if subset[j1][indexB] != partBs[i]:
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
    
            # if find no partA in the subset, create a new subset
            elif not found:
                # Each row represents a new person. The former 18 elements represents the global peak index.
                # Since when created, only one connection is added,
                # and each connection only have two peaks related,
                # there are only two values in the former 18 which are not -1
                # The 19-th one is the total score.
                # The 20-th one is set the parts already found for this person, should be 2 when created.
                row = -1 * np.ones(20)
                row[indexA] = partAs[i]
                row[indexB] = partBs[i]
                row[-1] = 2
                # caluclation of the score, i.e., row[-2]:
                #   the former is the heat map values sum of the start and end part
                #   the latter is the connection score calulated from paf
                # thus, row[-2] represents the total score of this connection
                row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                subset = np.vstack([subset, row])


    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    return candidate, all_peaks, subset

def visualize(img, candidate, all_peaks, subset, save_img_path=None):
    # visualize 1
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    
    canvas = np.copy(img) # B,G,R order
    
    for i in range(18):
        rgba = np.array(cmap(1 - i/18. - 1./36))
        rgba[0:3] *= 255
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
        to_plot = cv2.addWeighted(img, 0.3, canvas, 0.7, 0)
        '''
        cv2.imwrite('temp.jpg', to_plot)
        import pdb
        pdb.set_trace()
        '''
 
    to_plot = cv2.addWeighted(img, 0.3, canvas, 0.7, 0)
    # cv2.imwrite('part.jpg', to_plot)

    # visualize 2
    stickwidth = 4

    for i in range(19):
        # ignore the left/right shoulder to left/right ear connection
        if i == 9 or i == 13:
            continue
        for n in range(len(subset)):
            index = subset[n][np.array(cfg.limb_seq[i])]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i % len(colors)])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        '''
        cv2.imwrite('temp.jpg', canvas)
        import pdb
        pdb.set_trace()
        '''

    file_path = 'final.jpg' if save_img_path is None else save_img_path 
    cv2.imwrite(file_path, canvas)
    return canvas

if __name__ == '__main__':

    # img_id = 196283
    # img_id = 163640
    # img_id = 262148
    # img_path = os.path.join('coco/train2017', '%012d.jpg' % img_id)
    img_path = 'rgb_imgs/person_0/img_0.jpg'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model', required = True)
    parser.add_argument('--input_path', help='path of input data', default=img_path)
    args = parser.parse_args()

    predict_func = initialize(args.model_path)
    img = cv2.imread(args.input_path)
    candidate, all_peaks, subset = detect(img, predict_func)
    visualize(img, candidate, all_peaks, subset)
