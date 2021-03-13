# --------------------------------------------------------

# --------------------------------------------------------

"""
Change the HICO-DET detection results to the right format.
"""

import os
import pickle
import time

import numpy as np
import scipy.io as sio


pair_total_num = 999999

def save_HICO(HICO, HICO_dir, classid, begin, finish, fuse_type='spho'):
    all_boxes = []
    for i in range(finish - begin + 1):
        total = []
        score = []
        for key, value in HICO.items():
            for element in value:
                if element[2] == classid:
                    temp = []
                    temp.append(element[0].tolist())  # Human box
                    temp.append(element[1].tolist())  # Object box
                    temp.append(int(key))  # image id
                    temp.append(int(i))  # action id (0-599)

                    pSp = element[8]
                    pHOIs = element[9]

                    preds = pSp * pHOIs

                    temp.append(preds[begin - 1 + i] * element[4] * element[5])
                    total.append(temp)
                    score.append(preds[begin - 1 + i] * element[4] * element[5])

        idx = np.argsort(score, axis=0)[::-1]
        for i_idx in range(min(len(idx), 19999)):
            all_boxes.append(total[idx[i_idx]])
    savefile = HICO_dir + 'detections_' + str(classid).zfill(2) + '.mat'
    # print('length:', classid, len(all_boxes))
    sio.savemat(savefile, {'all_boxes': all_boxes})
    return all_boxes

def Generate_HICO_detection(output_file, HICO_dir, fuse_type, gpool):

    if not os.path.exists(HICO_dir):
        os.makedirs(HICO_dir)

    # Remove previous results
    filelist = [ f for f in os.listdir(HICO_dir)]
    for f in filelist:
        os.remove(os.path.join(HICO_dir, f))


    params = [[1 ,161, 170], # 1 person
              [2 ,11,  24],# 2 bicycle
              [3, 66, 76 ],  # 3 car
    [ 4, 147, 160],  # 4 motorcycle
    [ 5, 1, 10],  # 5 airplane
    [ 6, 55, 65],  # 6 bus
    [ 7, 187, 194],  # 7 train
    [ 8, 568, 576],  # 8 truck
    [ 9, 32, 46],  # 9 boat
    [ 10, 563, 567],  # 10 traffic light
    [ 11, 326, 330],  # 11 fire_hydrant
    [ 12, 503, 506],  # 12 stop_sign
    [ 13, 415, 418],  # 13 parking_meter
    [ 14, 244, 247],  # 14 bench
    [ 15, 25, 31],  # 15 bird
    [ 16, 77, 86],  # 16 cat
    [ 17, 112, 129],  # 17 dog
    [ 18, 130, 146],  # 18 horse
    [ 19, 175, 186],  # 19 sheep
    [ 20, 97, 107],  # 20 cow
    [ 21, 314, 325],  # 21 elephant
    [ 22, 236, 239],  # 22 bear
    [ 23, 596, 600],  # 23 zebra
    [ 24, 343, 348],  # 24 giraffe
    [ 25, 209, 214],  # 25 backpack
    [ 26, 577, 584],  # 26 umbrella
    [ 27, 353, 356],  # 27 handbag
    [ 28, 539, 546],  # 28 tie
    [ 29, 507, 516],  # 29 suitcase
    [ 30, 337, 342],  # 30 Frisbee
    [ 31, 464, 474],  # 31 skis
    [ 32, 475, 483],  # 32 snowboard
    [ 33, 489, 502],  # 33 sports_ball
    [ 34, 369, 376],  # 34 kite
    [ 35, 225, 232],  # 35 baseball_bat
    [ 36, 233, 235],  # 36 baseball_glove
    [ 37, 454, 463],  # 37 skateboard
    [ 38, 517, 528],  # 38 surfboard
    [ 39, 534, 538],  # 39 tennis_racket
    [ 40, 47, 54],  # 40 bottle
    [ 41, 589, 595],  # 41 wine_glass
    [ 42, 296, 305],  # 42 cup
    [ 43, 331, 336],  # 43 fork
    [ 44, 377, 383],  # 44 knife
    [ 45, 484, 488],  # 45 spoon
    [ 46, 253, 257],  # 46 bowl
    [ 47, 215, 224],  # 47 banana
    [ 48, 199, 208],  # 48 apple
    [ 49, 439, 445],  # 49 sandwich
    [ 50, 398, 407],  # 50 orange
    [ 51, 258, 264],  # 51 broccoli
    [ 52, 274, 283],  # 52 carrot
    [ 53, 357, 363],  # 53 hot_dog
    [ 54, 419, 429],  # 54 pizza
    [ 55, 306, 313],  # 55 donut
    [ 56, 265, 273],  # 56 cake
    [ 57, 87, 92],  # 57 chair
    [ 58, 93, 96],  # 58 couch
    [ 59, 171, 174],  # 59 potted_plant
    [ 60, 240, 243],  # 60 bed
    [ 61, 108, 111],  # 61 dining_table
    [ 62, 551, 558],  # 62 toilet
    [ 63, 195, 198],  # 63 TV
    [ 64, 384, 389],  # 64 laptop
    [ 65, 394, 397],  # 65 mouse
    [ 66, 435, 438],  # 66 remote
    [ 67, 364, 368],  # 67 keyboard
    [ 68, 284, 290],  # 68 cell_phone
    [ 69, 390, 393],  # 69 microwave
    [ 70, 408, 414],  # 70 oven
    [ 71, 547, 550],  # 71 toaster
    [ 72, 450, 453],  # 72 sink
    [ 73, 430, 434],  # 73 refrigerator
    [ 74, 248, 252],  # 74 book
    [ 75, 291, 295],  # 75 clock
    [ 76, 585, 588],  # 76 vase
    [ 77, 446, 449],  # 77 scissors
    [ 78, 529, 533],  # 78 teddy_bear
    [ 79, 349, 352],  # 79 hair_drier
    [ 80, 559, 562],  # 80 toothbrush

              ]

    import datetime

    # from multiprocessing import Pool
    #
    # process_num = 16 if fuse_type == 'spv' else 2
    # # global pool
    # # if pool is None:
    # pool = Pool(processes=process_num)
    # def func(item):
    #
    #     save_HICO(HICO, HICO_dir, item[0], item[1], item[2])
    #

    # gpool.starmap(save_HICO1, zip(repeat(output_file), repeat(HICO_dir), params, repeat(fuse_type)))

    from sys import version_info
    if version_info.major == 3:
        HICO = pickle.load(open(output_file, "rb"), encoding='latin1')
    else:
        HICO = pickle.load(open(output_file, "rb"))
    print('Load HICO sucessfully', datetime.datetime.now())
    for p in params:
        # print(p)
        save_HICO(HICO, HICO_dir, p[0], p[1], p[2], fuse_type)
        # print('end', p)
    # pool.close()
    # pool.join()
    # pool.terminate()
    # del pool
    # import gc
    # gc.collect()
    # pool.map(save_HICO, params)
    # save_HICO(HICO, HICO_dir,  1 ,161, 170) # 1 person
    # save_HICO(HICO, HICO_dir,  2 ,11,  24 ) # 2 bicycle
    # save_HICO(HICO, HICO_dir,  3 ,66,  76 ) # 3 car
    # save_HICO(HICO, HICO_dir,  4 ,147, 160) # 4 motorcycle
    # save_HICO(HICO, HICO_dir,  5 ,1,   10 ) # 5 airplane
    # save_HICO(HICO, HICO_dir,  6 ,55,  65 ) # 6 bus
    # save_HICO(HICO, HICO_dir,  7 ,187, 194) # 7 train
    # save_HICO(HICO, HICO_dir,  8 ,568, 576) # 8 truck
    # save_HICO(HICO, HICO_dir,  9 ,32,  46 ) # 9 boat
    # save_HICO(HICO, HICO_dir,  10,563, 567) # 10 traffic light
    # save_HICO(HICO, HICO_dir,  11,326,330) # 11 fire_hydrant
    # save_HICO(HICO, HICO_dir,  12,503,506) # 12 stop_sign
    # save_HICO(HICO, HICO_dir,  13,415,418) # 13 parking_meter
    # save_HICO(HICO, HICO_dir,  14,244,247) # 14 bench
    # save_HICO(HICO, HICO_dir,  15,25,  31) # 15 bird
    # save_HICO(HICO, HICO_dir,  16,77,  86) # 16 cat
    # save_HICO(HICO, HICO_dir,  17,112,129) # 17 dog
    # save_HICO(HICO, HICO_dir,  18,130,146) # 18 horse
    # save_HICO(HICO, HICO_dir,  19,175,186) # 19 sheep
    # save_HICO(HICO, HICO_dir,  20,97,107)  # 20 cow
    # save_HICO(HICO, HICO_dir,  21,314,325) # 21 elephant
    # save_HICO(HICO, HICO_dir,  22,236,239) # 22 bear
    # save_HICO(HICO, HICO_dir,  23,596,600) # 23 zebra
    # save_HICO(HICO, HICO_dir,  24,343,348) # 24 giraffe
    # save_HICO(HICO, HICO_dir,  25,209,214) # 25 backpack
    # save_HICO(HICO, HICO_dir,  26,577,584) # 26 umbrella
    # save_HICO(HICO, HICO_dir,  27,353,356) # 27 handbag
    # save_HICO(HICO, HICO_dir,  28,539,546) # 28 tie
    # save_HICO(HICO, HICO_dir,  29,507,516) # 29 suitcase
    # save_HICO(HICO, HICO_dir,  30,337,342) # 30 Frisbee
    # save_HICO(HICO, HICO_dir,  31,464,474) # 31 skis
    # save_HICO(HICO, HICO_dir,  32,475,483) # 32 snowboard
    # save_HICO(HICO, HICO_dir,  33,489,502) # 33 sports_ball
    # save_HICO(HICO, HICO_dir,  34,369,376) # 34 kite
    # save_HICO(HICO, HICO_dir,  35,225,232) # 35 baseball_bat
    # save_HICO(HICO, HICO_dir,  36,233,235) # 36 baseball_glove
    # save_HICO(HICO, HICO_dir,  37,454,463) # 37 skateboard
    # save_HICO(HICO, HICO_dir,  38,517,528) # 38 surfboard
    # save_HICO(HICO, HICO_dir,  39,534,538) # 39 tennis_racket
    # save_HICO(HICO, HICO_dir,  40,47,54)   # 40 bottle
    # save_HICO(HICO, HICO_dir,  41,589,595) # 41 wine_glass
    # save_HICO(HICO, HICO_dir,  42,296,305) # 42 cup
    # save_HICO(HICO, HICO_dir,  43,331,336) # 43 fork
    # save_HICO(HICO, HICO_dir,  44,377,383) # 44 knife
    # save_HICO(HICO, HICO_dir,  45,484,488) # 45 spoon
    # save_HICO(HICO, HICO_dir,  46,253,257) # 46 bowl
    # save_HICO(HICO, HICO_dir,  47,215,224) # 47 banana
    # save_HICO(HICO, HICO_dir,  48,199,208) # 48 apple
    # save_HICO(HICO, HICO_dir,  49,439,445) # 49 sandwich
    # save_HICO(HICO, HICO_dir,  50,398,407) # 50 orange
    # save_HICO(HICO, HICO_dir,  51,258,264) # 51 broccoli
    # save_HICO(HICO, HICO_dir,  52,274,283) # 52 carrot
    # save_HICO(HICO, HICO_dir,  53,357,363) # 53 hot_dog
    # save_HICO(HICO, HICO_dir,  54,419,429) # 54 pizza
    # save_HICO(HICO, HICO_dir,  55,306,313) # 55 donut
    # save_HICO(HICO, HICO_dir,  56,265,273) # 56 cake
    # save_HICO(HICO, HICO_dir,  57,87,92)   # 57 chair
    # save_HICO(HICO, HICO_dir,  58,93,96)   # 58 couch
    # save_HICO(HICO, HICO_dir,  59,171,174) # 59 potted_plant
    # save_HICO(HICO, HICO_dir,  60,240,243) #60 bed
    # save_HICO(HICO, HICO_dir,  61,108,111) #61 dining_table
    # save_HICO(HICO, HICO_dir,  62,551,558) #62 toilet
    # save_HICO(HICO, HICO_dir,  63,195,198) #63 TV
    # save_HICO(HICO, HICO_dir,  64,384,389) #64 laptop
    # save_HICO(HICO, HICO_dir,  65,394,397) #65 mouse
    # save_HICO(HICO, HICO_dir,  66,435,438) #66 remote
    # save_HICO(HICO, HICO_dir,  67,364,368) #67 keyboard
    # save_HICO(HICO, HICO_dir,  68,284,290) #68 cell_phone
    # save_HICO(HICO, HICO_dir,  69,390,393) #69 microwave
    # save_HICO(HICO, HICO_dir,  70,408,414) #70 oven
    # save_HICO(HICO, HICO_dir,  71,547,550) #71 toaster
    # save_HICO(HICO, HICO_dir,  72,450,453) #72 sink
    # save_HICO(HICO, HICO_dir,  73,430,434) #73 refrigerator
    # save_HICO(HICO, HICO_dir,  74,248,252) #74 book
    # save_HICO(HICO, HICO_dir,  75,291,295) #75 clock
    # save_HICO(HICO, HICO_dir,  76,585,588) #76 vase
    # save_HICO(HICO, HICO_dir,  77,446,449) #77 scissors
    # save_HICO(HICO, HICO_dir,  78,529,533) #78 teddy_bear
    # save_HICO(HICO, HICO_dir,  79,349,352) #79 hair_drier
    # save_HICO(HICO, HICO_dir,  80,559,562) #80 toothbrush
    print("Finish save HICO", datetime.datetime.now())
