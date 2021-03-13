# --------------------------------------------------------

# --------------------------------------------------------

"""
Generating training instance
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import random
from functools import partial
from random import randint

import cv2
import numpy as np
import tensorflow as tf

from .config import cfg

MAX_COCO_ID = 650000
MAX_HICO_ID = 40000

def bbox_trans(human_box_ori, object_box_ori, ratio, size = 64):

    human_box  = human_box_ori.copy()
    object_box = object_box_ori.copy()
    
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]    

    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    
    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'
        
    # shift the top-left corner to (0,0)
    
    human_box[0] -= InteractionPattern[0]
    human_box[2] -= InteractionPattern[0]
    human_box[1] -= InteractionPattern[1]
    human_box[3] -= InteractionPattern[1]    
    object_box[0] -= InteractionPattern[0]
    object_box[2] -= InteractionPattern[0]
    object_box[1] -= InteractionPattern[1]
    object_box[3] -= InteractionPattern[1] 

    if ratio == 'height': # height is larger than width
        
        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width  - 1 - human_box[2]) / height
        human_box[3] = (size - 1)                  - size * (height - 1 - human_box[3]) / height

        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width  - 1 - object_box[2]) / height
        object_box[3] = (size - 1)                  - size * (height - 1 - object_box[3]) / height
        
        # Need to shift horizontally  
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        #assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[3] == 63) & (InteractionPattern[2] <= 63)
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1



        shift = size / 2 - (InteractionPattern[2] + 1) / 2 
        human_box += [shift, 0 , shift, 0]
        object_box += [shift, 0 , shift, 0]
     
    else: # width is larger than height

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1)                  - size * (width  - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width
        

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1)                  - size * (width  - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width
        
        # Need to shift vertically 
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        
        
        #assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[2] == 63) & (InteractionPattern[3] <= 63)
        

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (InteractionPattern[3] + 1) / 2 
        
        human_box = human_box + [0, shift, 0 , shift]
        object_box = object_box + [0, shift, 0 , shift]

 
    return np.round(human_box), np.round(object_box)



def Get_next_sp(human_box, object_box, pattern_type=0):
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O  = bbox_trans(human_box, object_box, 'width')
    
    Pattern = np.zeros((64,64,2))
    obj_value = 1
    if pattern_type == 2:
        obj_value = 0.5
    elif pattern_type == 3:
        obj_value = -1
    Pattern[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 1
    Pattern[int(O[1]):int(O[3]) + 1,int(O[0]):int(O[2]) + 1,1] = obj_value


    return Pattern


def bb_IOU(boxA, boxB):

    ixmin = np.maximum(boxA[0], boxB[0])
    iymin = np.maximum(boxA[1], boxB[1])
    ixmax = np.minimum(boxA[2], boxB[2])
    iymax = np.minimum(boxA[3], boxB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
           (boxA[2] - boxA[0] + 1.) *
           (boxA[3] - boxA[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps


def Augmented_neg_box(bbox, shape, image_id, augment=15, bbox_list=[]):
    thres_ = 0.25

    # box = np.array([0, bbox[0], bbox[1], bbox[2], bbox[3]]).reshape(1, 5)
    # box = box.astype(np.float64)
    box = np.empty([1, 5], np.float64)

    count = 0
    time_count = 0
    while count < augment:

        time_count += 1
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]

        height_cen = (bbox[3] + bbox[1]) / 2
        width_cen = (bbox[2] + bbox[0]) / 2

        ratio = 1 + randint(-10, 10) * 0.01

        height_shift = randint(-np.floor(height), np.floor(height))
        height_shift = np.sign(height_shift)* 0.5 * height + height_shift
        width_shift = randint(-np.floor(width), np.floor(width)) * 0.1
        width_shift = np.sign(width_shift)* 0.5 * width + width_shift

        H_0 = max(0, width_cen + width_shift - ratio * width / 2)
        H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
        H_1 = max(0, height_cen + height_shift - ratio * height / 2)
        H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)

        valid_neg_box = True
        for bbox1 in bbox_list:
            if bb_IOU(bbox1, np.array([H_0, H_1, H_2, H_3])) > thres_:
                valid_neg_box = False
                break
        if valid_neg_box:
            box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1, 5)
            box = np.concatenate((box, box_), axis=0)
            count += 1
        if time_count > 150:
            return box

    return box


def Augmented_box(bbox, shape, image_id, augment = 15):

    thres_ = 0.7

    box = np.array([0, bbox[0],  bbox[1],  bbox[2],  bbox[3]]).reshape(1,5)
    box = box.astype(np.float64)
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return box
    count = 0
    time_count = 0
    while count < augment:
        
        time_count += 1
        height = bbox[3] - bbox[1]
        width  = bbox[2] - bbox[0]

        height_cen = (bbox[3] + bbox[1]) / 2
        width_cen  = (bbox[2] + bbox[0]) / 2

        ratio = 1 + randint(-10,10) * 0.01

        height_shift = randint(-np.floor(height),np.floor(height)) * 0.1
        width_shift  = randint(-np.floor(width),np.floor(width)) * 0.1

        H_0 = max(0, width_cen + width_shift - ratio * width / 2)
        H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
        H_1 = max(0, height_cen + height_shift - ratio * height / 2)
        H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)
        
        
        if bb_IOU(bbox, np.array([H_0, H_1, H_2, H_3])) > thres_:
            box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1,5)
            box  = np.concatenate((box,     box_),     axis=0)
            count += 1
        if time_count > 150:
            return box
            
    return box


def Generate_action_HICO(action_list):
    action_ = np.zeros(600)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,600)
    return action_

def Augmented_HO_Neg_HICO(GT, Trainval_Neg, shape, Pos_augment, Neg_select, pattern_type=False, isalign=False, box_list=[],
                          real_neg_ratio=0):
    """

    :param GT:
    :param Trainval_Neg:
    :param shape:
    :param Pos_augment:
    :param Neg_select:
    :param pattern_type:
    :param isalign:
    :param box_list:
    :param real_neg_ratio: This is for no action HOI (all zeros)
    :return:
    """
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]

    action_HO_ = Generate_action_HICO(GT[1])
    action_HO  = action_HO_

    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
    if isalign:
        while len(Human_augmented) < Pos_augment + 1:
            Human_augmented = np.concatenate([Human_augmented, Human_augmented[-(Pos_augment + 1 - len(Human_augmented)):]], axis=0)

    if isalign:
        while len(Object_augmented) < Pos_augment + 1:
            Object_augmented = np.concatenate([Object_augmented, Object_augmented[-(Pos_augment + 1 - len(Human_augmented)):]], axis=0)
    # print("shape:", Human_augmented.shape, Object_augmented.shape)
    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]
    # print("shape1:", Human_augmented.shape, Object_augmented.shape)
    if isalign:
        assert len(Human_augmented) == Pos_augment + 1, (len(Human_augmented), Pos_augment)

    action_HO = np.tile(action_HO, [len(Human_augmented), 1])
    # for i in range(len(Human_augmented) - 1):
    #     action_HO = np.concatenate((action_HO, action_HO_), axis=0)

    if len(box_list) > 0 and real_neg_ratio > 0:
        aug_neg_objs = Augmented_neg_box(Object, shape, image_id, int(Pos_augment * real_neg_ratio), bbox_list = box_list)
        if len(aug_neg_objs) > 0:
            aug_neg_humans = np.tile([Human_augmented[0]], [len(aug_neg_objs), 1])
            aug_neg_actions = np.zeros([len(aug_neg_objs), 600], )
            # print(aug_neg_objs.shape, Object_augmented.shape, Human_augmented.shape, aug_neg_humans.shape)
            Human_augmented = np.concatenate([Human_augmented, aug_neg_humans])
            Object_augmented = np.concatenate([Object_augmented, aug_neg_objs])
            action_HO = np.concatenate([action_HO, aug_neg_actions])

    num_pos = len(Human_augmented)
    if pattern_type == 1: pose_list = [GT[5]] * num_pos

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)

    num_pos_neg = len(Human_augmented)
    pattern_channel = 2
    Pattern   = np.empty((0, 64, 64, pattern_channel), dtype=np.float32)
    obj_mask = np.empty((0, shape[0] // 16, shape[1] // 16, 1), dtype=np.float32)
    mask_all = np.zeros(shape=(1, shape[0], shape[1], 1), dtype=np.float32)

    for i in range(num_pos_neg):
        # Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        # there are poses for the negative sample
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:], pattern_type)
        Pattern_ = Pattern_.reshape(1, 64, 64, pattern_channel)

        mask = np.zeros(shape=(1, shape[0], shape[1], 1), dtype=np.float32)
        obj_box = Object_augmented[i][1:].astype(np.int32)
        # print(obj_box)
        mask[:, obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]] = 1

        if i < num_pos: mask_all[:, obj_box[0]:obj_box[2], obj_box[1]:obj_box[3]] = 1

        from skimage import transform
        mask = transform.resize(mask, [1, shape[0] // 16, shape[1] // 16, 1], order=0, preserve_range=True)
        obj_mask = np.concatenate((obj_mask, mask), axis=0)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, pattern_channel)
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5)
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5)
    action_HO         = action_HO.reshape(num_pos_neg, 600)

    # print("shape1:", Human_augmented.shape, Object_augmented.shape, num_pos, Neg_select)
    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, obj_mask, mask_all



def get_new_Trainval_N(Trainval_N, is_zero_shot, unseen_idx):
    if is_zero_shot > 0:
        # remove zero-shot results.
        new_Trainval_N = {}
        for k in Trainval_N.keys():
            new_Trainval_N[k] = []
            for item in Trainval_N[4]:
                if item[1] not in unseen_idx:
                    new_Trainval_N[k].append(item)
        Trainval_N = new_Trainval_N
    return Trainval_N


def get_zero_shot_type(model_name):
    zero_shot_type = 0
    if model_name.__contains__('_zs_'):
        zero_shot_type = 7
    elif model_name.__contains__('zsnrare'):
        zero_shot_type = 4
    elif model_name.__contains__('_zsrare_'):
        zero_shot_type = 3
    elif model_name.__contains__('_zsuo_'):
        # for unseen object
        zero_shot_type = 11
    return zero_shot_type


def get_epoch_iters(model_name):
    epoch_iters = 43273
    if model_name.__contains__('zsnrare'):
        epoch_iters = 20000
    elif model_name.__contains__('zs_'):
        epoch_iters = 20000
    elif model_name.__contains__('zsrare'):
        epoch_iters = 40000
    else:
        epoch_iters = 43273
    return epoch_iters

def get_augment_type(model_name):
    augment_type = 4
    return augment_type


def get_unseen_index(zero_shot_type):
    unseen_idx = None
    if zero_shot_type == 3:
        # rare
        unseen_idx = [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418, 70, 416,
         389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596, 345, 189,
         205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229, 158, 195,
         238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188, 216, 597,
         77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104, 55, 50,
         198, 168, 391, 192, 595, 136, 581]

    elif zero_shot_type == 11:
        unseen_idx = [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                      126, 127, 128, 224, 225, 226, 227, 228, 229, 230, 231, 290, 291, 292, 293,
                      294, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 336, 337,
                      338, 339, 340, 341, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
                      429, 430, 431, 432, 433, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
                      463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 533, 534, 535, 536,
                      537, 558, 559, 560, 561, 595, 596, 597, 598, 599]
        #  miss [ 5, 6, 28, 56, 88] verbs 006  break    007  brush_with 029  flip  057  move  089  slide
    elif zero_shot_type == 4:
        # non rare
        unseen_idx = [38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75, 212, 472, 61,
         457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479, 230, 385, 73,
         159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338, 29, 594, 346,
         456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191, 266, 304, 6, 572,
         529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329, 246, 173, 506,
         383, 93, 516, 64]
        # 25729, 93041
    elif zero_shot_type == 7:
        # 24 rare merge of zsnrare & zsrare
        unseen_idx = [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418, 70, 416, 389,
         90, 38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75, 212, 472, 61,
         457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479, 230, 385, 73, 159,
         190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338, 29, 594, 346, 456, 589,
         45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191, 266, 304, 6, 572, 529, 312,
         9]
        # 22529, 14830, 22493, 17411, 21912,
    return unseen_idx

def generator(Trainval_GT, Trainval_N, Pos_augment, Neg_select, augment_type, pattern_type, zero_shot_type, isalign,
              epoch=0, neg_type=0, adj_matrix=False):
    """
    fir the problem of generator1
    :param neg_type:
    :param Trainval_GT:
    :param Trainval_N:
    :param Pos_augment:
    :param Neg_select:
    :param augment_type:
    :param pattern_type:
    :return:
    """
    # assert skimage.__version__ == '0.14.2'
    Neg_select1, Pos_augment1, inters_per_img = get_aug_params(Neg_select, Pos_augment, augment_type)
    unseen_idx = get_unseen_index(zero_shot_type)
    Trainval_N = get_new_Trainval_N(Trainval_N, zero_shot_type, unseen_idx)
    print("generator", inters_per_img, Pos_augment1, 'Neg_select:', Neg_select1, augment_type, 'zero shot:', zero_shot_type)

    import math
    img_id_index_map = {}
    for i, gt in enumerate(Trainval_GT):
        img_id = gt[0]
        if img_id in img_id_index_map:
            img_id_index_map[img_id].append(i)
        else:
            img_id_index_map[img_id] = [i]
    img_id_list = list(img_id_index_map.keys())
    for k, v in img_id_index_map.items():
        for i in range(math.ceil(len(v) * 1.0 / inters_per_img) - 1):
            img_id_list.append(k)
    import copy
    import time
    st = time.time()
    count_time = 0
    avg_time = 0
    while True:
        running_map = copy.deepcopy(img_id_index_map)
        if augment_type >= 0: # augment_type < 0 is for test
            np.random.shuffle(img_id_list)
            for k in running_map.keys():
                np.random.shuffle(running_map[k])

        for img_id_tmp in img_id_list:
            gt_ids = running_map[img_id_tmp][:inters_per_img]
            running_map[img_id_tmp] = running_map[img_id_tmp][inters_per_img:]
            Pattern_list = []
            Human_augmented_list = []
            Object_augmented_list = []
            action_HO_list = []
            num_pos_list = 0
            obj_mask_list = []
            mask_all_list = []

            image_id = img_id_tmp
            if image_id in [528, 791, 1453, 2783, 3489, 3946, 3946, 11747, 11978, 12677, 16946, 17833, 19218, 19218, 22347, 27293, 27584, 28514, 33683, 35399]:
                continue
            im_file = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (
                    str(image_id)).zfill(8) + '.jpg'
            # id, gt, h, o
            # print(gt_ids, gt_ids[0], Trainval_GT[gt_ids[0]])
            if Trainval_GT[gt_ids[0]][2] == [0., 0., 0., 0.]:
                if image_id >= MAX_COCO_ID:
                    # obj365
                    tmp_id = image_id - MAX_COCO_ID
                    im_file = cfg.LOCAL_DATA + '/dataset/Objects365/Images/train/train/obj365_train_' + (str(tmp_id)).zfill(
                        12) + '.jpg'
                    pass
                else:
                    tmp_id = image_id - MAX_HICO_ID
                    im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(tmp_id)).zfill(
                        12) + '.jpg'
                    import os
                    if not os.path.exists(im_file):
                        im_file = cfg.DATA_DIR + '/' + 'v-coco/coco/images/val2014/COCO_val2014_' + (
                            str(tmp_id)).zfill(12) + '.jpg'
                        if not os.path.exists(im_file):
                            print(im_file)
            import cv2
            import os
            if not os.path.exists(im_file):
                print('not exist', im_file)
                continue
            im = cv2.imread(im_file)
            if im is None:
                print('node', im_file)
                continue
            im_orig = im.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS
            im_shape = im.shape

            human_box_list = []
            obj_box_list = []
            for i in gt_ids:
                GT = Trainval_GT[i]
                # unseen data
                if zero_shot_type > 0:
                    has_unseen = False
                    for label in GT[1]:
                        if label in unseen_idx and GT[2] != [0., 0., 0., 0.]:
                            # remove unseen in HICO
                            has_unseen = True
                    if has_unseen:
                        continue
                assert GT[0] == image_id

                cur_pos_augment = Pos_augment1
                if augment_type > 1:
                    if i == gt_ids[-1]:   # This must be -1
                        cur_neg_select = Neg_select1 * len(gt_ids)
                    else:
                        cur_neg_select = 0
                else:
                    cur_neg_select = Neg_select1

                bbox_list = []

                Pattern, Human_augmented, Object_augmented, action_HO, num_pos, obj_mask, mask_all = Augmented_HO_Neg_HICO(
                    GT, Trainval_N, im_shape, Pos_augment=cur_pos_augment, Neg_select=cur_neg_select,
                    pattern_type=pattern_type, isalign=isalign, box_list=bbox_list, real_neg_ratio=neg_type)

                if adj_matrix:
                    human_box_list.append(np.tile([GT[2]], [len(Human_augmented), 1]) )
                    obj_box_list.append(np.tile([GT[3]], [len(Human_augmented), 1]))

                # maintain same number of augmentation,

                Pattern_list.append(Pattern)
                Human_augmented_list.append(Human_augmented)
                Object_augmented_list.append(Object_augmented)
                action_HO_list.append(action_HO)
                num_pos_list += num_pos
                obj_mask_list.append(obj_mask)
                mask_all_list.append(mask_all)
            if len(Pattern_list) <= 0:
                continue
            Pattern = np.concatenate(Pattern_list, axis=0)
            Human_augmented = np.concatenate(Human_augmented_list, axis=0)
            Object_augmented = np.concatenate(Object_augmented_list, axis=0)
            action_HO = np.concatenate(action_HO_list, axis=0)
            num_pos = num_pos_list
            obj_mask = np.concatenate(obj_mask_list, axis=0)
            # print(mask_all.shape)
            im_orig = np.expand_dims(im_orig, axis=0)
            # print('item:', num_pos, np.where(action_HO[num_pos:])[1])
            # import os
            # print(os.getpid(), np.random.rand())
            if adj_matrix:
                human_gt_box_list = np.concatenate(human_box_list, axis=0)
                obj_gt_box_list = np.concatenate(obj_box_list, axis=0)
                yield (im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern,
                       human_gt_box_list, obj_gt_box_list, obj_mask)
            else:
                yield (im_orig, image_id, num_pos, Human_augmented, Object_augmented, action_HO, Pattern, obj_mask)
        #     avg_time = ((time.time() - st) + avg_time * count_time) / (count_time + 1)
        #     count_time += 1
        #     print('generate batch:', time.time() - st, "average;", avg_time, count_time)
        #     st = time.time()
        # print(count_time,'hhh')
        # exit()
        if augment_type < 0:
            break




def get_aug_params(Neg_select, Pos_augment, augment_type):
    Pos_augment1 = Pos_augment
    Neg_select1 = Neg_select
    # increate the number of HOIs in each batch.
    inters_per_img = 5
    Pos_augment1 = 6
    Neg_select1 = 24
    return Neg_select1, Pos_augment1, inters_per_img


def obtain_data(Pos_augment=15, Neg_select=60, augment_type=0, pattern_type=False, zero_shot_type=0, isalign=False,
                epoch=0, coco=False, neg_type=0):
    with open(cfg.DATA_DIR + '/' + 'Trainval_GT_HICO.pkl', "rb") as f:
        Trainval_GT = pickle.load(f, encoding='latin1')
    with open(cfg.DATA_DIR + '/' + 'Trainval_Neg_HICO.pkl', "rb") as f:
        Trainval_N = pickle.load(f, encoding='latin1')

    if pattern_type == 1:
        pattern_channel = 3
    else:
        pattern_channel = 2
    dataset = tf.data.Dataset.from_generator(partial(generator, Trainval_GT, Trainval_N, Pos_augment, Neg_select,
                                                     augment_type, pattern_type, zero_shot_type, isalign, epoch, neg_type), output_types=(
        tf.float32, tf.int32, tf.int64, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32),
                                             output_shapes=(
                                             tf.TensorShape([1, None, None, 3]), tf.TensorShape([]), tf.TensorShape([]),
                                             tf.TensorShape([None, 5]), tf.TensorShape([None, 5]),
                                             tf.TensorShape([None, 600]),
                                             tf.TensorShape([None, 64, 64, pattern_channel]),
                                             tf.TensorShape([None, None, None, 1])))
    # dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32),
    #                                          output_shapes=(tf.TensorShape([1, None, None, 3]), tf.TensorShape([])))
    dataset = dataset.prefetch(100)
    # dataset = dataset.shuffle(1000)
    # dataset = dataset.repeat(100)
    # dataset = dataset.repeat(1000).shuffle(1000)
    # dataset._dataset.batch(3)
    iterator = dataset.make_one_shot_iterator()
    image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, obj_mask = iterator.get_next()
    return image, image_id, num_pos, Human_augmented, Object_augmented, action_HO, sp, obj_mask


