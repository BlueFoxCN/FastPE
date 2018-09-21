from easydict import EasyDict as edict

_ = cfg = edict()



_.num_deconv_layers = 3

_.num_deconv_filters = [256, 256, 256]

_.num_deconv_kernels = [4, 4, 4]

_.deconv_with_bias = False

_.final_num_joints = 17

_.final_conv_kernel = 1

_.init_weights = False

_.use_target_weight = True

_.weight_decay = 5e-4

_.base_lr = 1e-3

_.depth = 50

_.momentum = 0.9

_.pixel_std = 200

_.image_size = [192, 256]# width * height

_.heatmap_size = [48, 64] # width * height, ex: 24 * 32

_.use_gt_bbox = True

_.select_data = False

_.data_format = 'jpg'

_.scale_factor = 0.3

_.rot_factor = 40

_.flip = True

_.sigma = 2

_.is_bgr = True

_.target_type = 'gaussian'

_.train_ann = 'coco/annotations/person_keypoints_train2017.json'
_.val_ann = 'coco/annotations/person_keypoints_val2017.json'

_.train_labels_dir = 'coco/train2017_labels'
_.val_labels_dir = 'coco/val2017_labels'




# cfg.backbone = 'vgg19'



# _.train_images_dir = 'coco/train2017'
# _.val_images_dir = 'coco/val2017'
# _.test_images_dir = 'coco/test2017'

# _.train_masks_dir = 'coco/train2017_masks'
# _.val_masks_dir = 'coco/val2017_masks'



# _.img_y = 368
# _.img_x = 368

# # whether skip a person if the distance to existing person is too small
# _.skip_adj = False

# _.augmentation = True
# _.bias_lr_mult = 2
# _.lr_mult = 4

# _.debug = False
# _.debug_sample_num = 1600

# # the default oder in coco annotation is:
# # 0:nose          1:left_eye         2:right_eye         3:left_ear
# # 4:right_ear     5:left_shoulder    6:right_shoulder    7:left_elbow
# # 8:right_elbow   9:left_wrist       10:right_wrist      11:left_hip
# # 12:right_hip    13:left_knee       14:right_knee       15:left_ankle
# # 16:right_ankle
# _.from_body_part = [0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
# _.to_body_part =   [0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
# # the oder after adjustment is:
# # 0:nose         1:neck           2:right_shoulder   3:right_elbow
# # 4:right_wrist  5:left_shoulder  6:left_elbow       7:left_wrist
# # 8:right_hip    9:right_knee     10:right_anckle    11:left_hip
# # 12:left_knee   13:left_ankle    14:right_eye       15:left_eye
# # 16:right_ear   17:left_ear

# _.coco_to_ours = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

# # the limbs include:
# # 0:neck-->right_hip           1:right_hip-->right_knee      2:right_knee-->right_ankle      3:neck-->left_hip
# # 4:left_hip-->left_knee       5:left_knee-->left_ankle      6:neck-->right_shoulder         7:right_shoulder-->right_elbow
# # 8:right_elbow-->right_wrist  9:right_shoulder-->right_ear  10:neck-->left_shoulder         11:left_shoulder-->left_elbow
# # 12:left_elbow-->left_wrist   13:left_shoulder-->left_ear   14:neck-->nose                  15:nose-->right_eye
# # 16:nose-->left_eye           17:right_eye-->right_ear      18:left_eye-->left_ear
# _.limb_from = [1, 8,  9,  1, 11, 12, 1, 2, 3,  2, 1, 5, 6,  5, 1,  0,  0, 14, 15]
# _.limb_to =   [8, 9, 10, 11, 12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17]

# _.limb_seq = []
# _.map_idx = []
# for idx in range(len(_.limb_from)):
#     _.limb_seq.append([_.limb_from[idx], _.limb_to[idx]])
#     _.map_idx.append([2 * idx, 2 * idx + 1])

# _.stride = 8



# _.grid_y = int(_.img_y / _.stride)
# _.grid_x = int(_.img_x / _.stride)

# _.thre = 1
# _.thre1 = 0.1
# _.thre2 = 0.05

# _.ch_heats = 18 + 1 # 18个parts 第19个(遍历不会执行到)表示是非人体区域
# _.ch_vectors = 38
# _.stages = 6

# _.scale_search = [0.5, 1, 1.5, 2]
# _.pad_value = 128


