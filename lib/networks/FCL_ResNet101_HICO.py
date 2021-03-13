

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.python.framework import ops

from networks.Fabricator import Fabricator
from networks.tools import get_convert_matrix
from ult.config import cfg


def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }
    with arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
        weights_initializer = slim.variance_scaling_initializer(),
        biases_regularizer  = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), 
        biases_initializer  = tf.constant_initializer(0.0),
        trainable           = is_training,
        activation_fn       = tf.nn.relu,
        normalizer_fn       = slim.batch_norm,
        normalizer_params   = batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc



class ResNet101():
    def __init__(self, model_name):
        self.model_name = model_name
        self.visualize = {}
        self.test_visualize = {}
        self.predictions = {}
        self.score_summaries = {}
        self.event_summaries = {}
        self.train_summaries = []
        self.losses = {}


        self.gt_class_HO_for_G_verbs = None
        self.gt_class_HO_for_D_verbs = None


        self.losses['fake_D_total_loss'] = 0
        self.losses['fake_G_total_loss'] = 0
        self.losses['fake_total_loss'] = 0
        self.update_ops = []

        self.image       = tf.placeholder(tf.float32, shape=[1, None, None, 3], name = 'image')
        self.spatial     = tf.placeholder(tf.float32, shape=[None, 64, 64, 2], name = 'sp')
        self.H_boxes     = tf.placeholder(tf.float32, shape=[None, 5], name = 'H_boxes')
        self.O_boxes     = tf.placeholder(tf.float32, shape=[None, 5], name = 'O_boxes')
        self.gt_class_HO = tf.placeholder(tf.float32, shape=[None, 600], name = 'gt_class_HO')
        self.H_num       = tf.placeholder(tf.int32)    # positive nums
        self.image_id    = tf.placeholder(tf.int32)
        self.num_classes = 600
        self.compose_num_classes = 600
        self.num_fc      = 1024
        self.verb_num_classes = 117
        self.obj_num_classes = 80
        self.scope       = 'resnet_v1_101'
        self.stride      = [16, ]
        self.lr          = tf.placeholder(tf.float32)
        if tf.__version__ == '1.1.0':
            raise Exception('wrong tensorflow version 1.1.0')
        else:
            from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
            self.blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                            resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                            resnet_v1_block('block3', base_depth=256, num_units=23, stride=1),
                            resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
                            resnet_v1_block('block5', base_depth=512, num_units=3, stride=1)]
        # this is from TIN
        self.HO_weight = np.array([
            9.192927, 9.778443, 10.338059, 9.164914, 9.075144, 10.045923, 8.714437, 8.59822, 12.977117, 6.2745423,
            11.227917, 6.765012, 9.436157, 9.56762, 11.0675745, 11.530198, 9.609821, 9.897503, 6.664475, 6.811699,
            6.644726, 9.170454, 13.670264, 3.903943, 10.556748, 8.814335, 9.519224, 12.753973, 11.590822, 8.278912,
            5.5245695, 9.7286825, 8.997436, 10.699849, 9.601237, 11.965516, 9.192927, 10.220277, 6.056692, 7.734048,
            8.42324, 6.586457, 6.969533, 10.579222, 13.670264, 4.4531965, 9.326459, 9.288238, 8.071842, 10.431585,
            12.417501, 11.530198, 11.227917, 4.0678477, 8.854023, 12.571651, 8.225684, 10.996116, 11.0675745,
            10.100731,
            7.0376034, 7.463688, 12.571651, 14.363411, 5.4902234, 11.0675745, 14.363411, 8.45805, 10.269067,
            9.820116,
            14.363411, 11.272368, 11.105314, 7.981595, 9.198626, 3.3284247, 14.363411, 12.977117, 9.300817,
            10.032678,
            12.571651, 10.114916, 10.471591, 13.264799, 14.363411, 8.01953, 10.412168, 9.644913, 9.981384,
            7.2197933,
            14.363411, 3.1178555, 11.031207, 8.934066, 7.546675, 6.386472, 12.060826, 8.862153, 9.799063, 12.753973,
            12.753973, 10.412168, 10.8976755, 10.471591, 12.571651, 9.519224, 6.207762, 12.753973, 6.60636,
            6.2896967,
            4.5198326, 9.7887, 13.670264, 11.878505, 11.965516, 8.576513, 11.105314, 9.192927, 11.47304, 11.367679,
            9.275815, 11.367679, 9.944571, 11.590822, 10.451388, 9.511381, 11.144535, 13.264799, 5.888291,
            11.227917,
            10.779892, 7.643191, 11.105314, 9.414651, 11.965516, 14.363411, 12.28397, 9.909063, 8.94731, 7.0330057,
            8.129001, 7.2817025, 9.874775, 9.758241, 11.105314, 5.0690055, 7.4768796, 10.129305, 9.54313, 13.264799,
            9.699972, 11.878505, 8.260853, 7.1437693, 6.9321113, 6.990665, 8.8104515, 11.655361, 13.264799,
            4.515912,
            9.897503, 11.418972, 8.113436, 8.795067, 10.236277, 12.753973, 14.363411, 9.352776, 12.417501,
            0.6271591,
            12.060826, 12.060826, 12.166186, 5.2946343, 11.318889, 9.8308115, 8.016022, 9.198626, 10.8976755,
            13.670264,
            11.105314, 14.363411, 9.653881, 9.503599, 12.753973, 5.80546, 9.653881, 9.592727, 12.977117, 13.670264,
            7.995224, 8.639826, 12.28397, 6.586876, 10.929424, 13.264799, 8.94731, 6.1026597, 12.417501, 11.47304,
            10.451388, 8.95624, 10.996116, 11.144535, 11.031207, 13.670264, 13.670264, 6.397866, 7.513285, 9.981384,
            11.367679, 11.590822, 7.4348736, 4.415428, 12.166186, 8.573451, 12.977117, 9.609821, 8.601359, 9.055143,
            11.965516, 11.105314, 13.264799, 5.8201604, 10.451388, 9.944571, 7.7855496, 14.363411, 8.5463,
            13.670264,
            7.9288645, 5.7561946, 9.075144, 9.0701065, 5.6871653, 11.318889, 10.252538, 9.758241, 9.407584,
            13.670264,
            8.570397, 9.326459, 7.488179, 11.798462, 9.897503, 6.7530537, 4.7828183, 9.519224, 7.6492405, 8.031909,
            7.8180614, 4.451856, 10.045923, 10.83705, 13.264799, 13.670264, 4.5245686, 14.363411, 10.556748,
            10.556748,
            14.363411, 13.670264, 14.363411, 8.037262, 8.59197, 9.738439, 8.652985, 10.045923, 9.400566, 10.9622135,
            11.965516, 10.032678, 5.9017305, 9.738439, 12.977117, 11.105314, 10.725825, 9.080208, 11.272368,
            14.363411,
            14.363411, 13.264799, 6.9279733, 9.153925, 8.075553, 9.126969, 14.363411, 8.903826, 9.488214, 5.4571533,
            10.129305, 10.579222, 12.571651, 11.965516, 6.237189, 9.428937, 9.618479, 8.620408, 11.590822,
            11.655361,
            9.968962, 10.8080635, 10.431585, 14.363411, 3.796231, 12.060826, 10.302968, 9.551227, 8.75394,
            10.579222,
            9.944571, 14.363411, 6.272396, 10.625742, 9.690582, 13.670264, 11.798462, 13.670264, 11.724354,
            9.993963,
            8.230013, 9.100721, 10.374427, 7.865129, 6.514087, 14.363411, 11.031207, 11.655361, 12.166186, 7.419324,
            9.421769, 9.653881, 10.996116, 12.571651, 13.670264, 5.912144, 9.7887, 8.585759, 8.272101, 11.530198,
            8.886948,
            5.9870906, 9.269661, 11.878505, 11.227917, 13.670264, 8.339964, 7.6763024, 10.471591, 10.451388,
            13.670264,
            11.185357, 10.032678, 9.313555, 12.571651, 3.993144, 9.379805, 9.609821, 14.363411, 9.709451, 8.965248,
            10.451388, 7.0609145, 10.579222, 13.264799, 10.49221, 8.978916, 7.124196, 10.602211, 8.9743395, 7.77862,
            8.073695, 9.644913, 9.339531, 8.272101, 4.794418, 9.016304, 8.012526, 10.674532, 14.363411, 7.995224,
            12.753973, 5.5157638, 8.934066, 10.779892, 7.930471, 11.724354, 8.85808, 5.9025764, 14.363411,
            12.753973,
            12.417501, 8.59197, 10.513264, 10.338059, 14.363411, 7.7079706, 14.363411, 13.264799, 13.264799,
            10.752493,
            14.363411, 14.363411, 13.264799, 12.417501, 13.670264, 6.5661197, 12.977117, 11.798462, 9.968962,
            12.753973,
            11.47304, 11.227917, 7.6763024, 10.779892, 11.185357, 14.363411, 7.369478, 14.363411, 9.944571,
            10.779892,
            10.471591, 9.54313, 9.148476, 10.285873, 10.412168, 12.753973, 14.363411, 6.0308623, 13.670264,
            10.725825,
            12.977117, 11.272368, 7.663911, 9.137665, 10.236277, 13.264799, 6.715625, 10.9622135, 14.363411,
            13.264799,
            9.575919, 9.080208, 11.878505, 7.1863923, 9.366199, 8.854023, 9.874775, 8.2857685, 13.670264, 11.878505,
            12.166186, 7.616999, 9.44343, 8.288065, 8.8104515, 8.347254, 7.4738197, 10.302968, 6.936267, 11.272368,
            7.058223, 5.0138307, 12.753973, 10.173757, 9.863602, 11.318889, 9.54313, 10.996116, 12.753973,
            7.8339925,
            7.569945, 7.4427395, 5.560738, 12.753973, 10.725825, 10.252538, 9.307165, 8.491293, 7.9161053,
            7.8849015,
            7.782772, 6.3088884, 8.866243, 9.8308115, 14.363411, 10.8976755, 5.908519, 10.269067, 9.176025,
            9.852551,
            9.488214, 8.90809, 8.537411, 9.653881, 8.662968, 11.965516, 10.143904, 14.363411, 14.363411, 9.407584,
            5.281472, 11.272368, 12.060826, 14.363411, 7.4135547, 8.920994, 9.618479, 8.891141, 14.363411,
            12.060826,
            11.965516, 10.9622135, 10.9622135, 14.363411, 5.658909, 8.934066, 12.571651, 8.614018, 11.655361,
            13.264799,
            10.996116, 13.670264, 8.965248, 9.326459, 11.144535, 14.363411, 6.0517673, 10.513264, 8.7430105,
            10.338059,
            13.264799, 6.878481, 9.065094, 8.87035, 14.363411, 9.92076, 6.5872955, 10.32036, 14.363411, 9.944571,
            11.798462, 10.9622135, 11.031207, 7.652888, 4.334878, 13.670264, 13.670264, 14.363411, 10.725825,
            12.417501,
            14.363411, 13.264799, 11.655361, 10.338059, 13.264799, 12.753973, 8.206432, 8.916674, 8.59509,
            14.363411,
            7.376845, 11.798462, 11.530198, 11.318889, 11.185357, 5.0664344, 11.185357, 9.372978, 10.471591,
            9.6629305,
            11.367679, 8.73579, 9.080208, 11.724354, 5.04781, 7.3777695, 7.065643, 12.571651, 11.724354, 12.166186,
            12.166186, 7.215852, 4.374113, 11.655361, 11.530198, 14.363411, 6.4993753, 11.031207, 8.344818,
            10.513264,
            10.032678, 14.363411, 14.363411, 4.5873594, 12.28397, 13.670264, 12.977117, 10.032678, 9.609821
        ], dtype='float32')
        self.HO_weight = self.HO_weight.reshape(1, 600)

        num_inst_path = cfg.ROOT_DIR +  '/Data/num_inst.npy'
        num_inst = np.load(num_inst_path)
        self.num_inst = num_inst
        self.num_inst_all = np.load(cfg.ROOT_DIR + '/Data/num_inst_new.npy')

        rare_criteria = 10  # or 100
        tmp = np.where(num_inst > rare_criteria)[0]
        tmp1 = np.zeros(self.num_classes)
        tmp1[tmp] = 1
        self.non_rare_cls_index = tf.constant(tmp1)

        tmp = np.where(num_inst <= rare_criteria)[0]
        tmp1 = np.zeros(self.num_classes)
        tmp1[tmp] = 1
        self.rare_cls_index = tf.constant(tmp1)

        verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix(self.verb_num_classes, self.obj_num_classes)
        verb_nums = np.matmul(self.num_inst, verb_to_HO_matrix.transpose())
        reweights = np.log(1 / (verb_nums / np.sum(verb_nums)))
        reweights = np.asarray(reweights, np.float32)
        self.verb_weights = np.expand_dims(reweights, axis=0)

        obj_nums = np.matmul(self.num_inst, obj_to_HO_matrix.transpose())
        reweights = np.log(1 / (obj_nums / np.sum(obj_nums)))
        reweights = np.asarray(reweights, np.float32)
        self.obj_weights = np.expand_dims(reweights, axis=0)

        self.verb_to_HO_matrix_np = np.asarray(verb_to_HO_matrix, np.float32)

        self.obj_to_HO_matrix = tf.constant(obj_to_HO_matrix, tf.float32)
        self.verb_to_HO_matrix = tf.constant(verb_to_HO_matrix, tf.float32)
        self.gt_obj_class = tf.cast(tf.matmul(self.gt_class_HO, self.obj_to_HO_matrix, transpose_b=True) > 0,
                                    tf.float32)
        self.gt_verb_class = tf.cast(tf.matmul(self.gt_class_HO, self.verb_to_HO_matrix, transpose_b=True) > 0,
                                     tf.float32)

        from networks.tools import get_word2vec
        word2vec = get_word2vec()
        self.word2vec_emb = tf.constant(word2vec)
        self.feature_gen = Fabricator(self)

    def set_gt_class_HO_for_G_verbs(self, gt_class_HO_for_G_verbs):
        self.gt_class_HO_for_G_verbs = gt_class_HO_for_G_verbs

    def set_gt_class_HO_for_D_verbs(self, gt_class_HO_for_D_verbs):
        self.gt_class_HO_for_D_verbs = gt_class_HO_for_D_verbs

    def init_table(self):
        pass

    def set_ph(self, image, image_id, num_pos, Human_augmented, Object_augmented, action_HO=None, sp=None):
        if image is not None: self.image = image
        if image_id is not None: self.image_id = image_id
        if sp is not None: self.spatial = sp
        if Human_augmented is not None: self.H_boxes = Human_augmented
        if Object_augmented is not None: self.O_boxes = Object_augmented
        if action_HO is not None: self.gt_class_HO = action_HO
        self.H_num = num_pos

        self.reset_classes()

    def reset_classes(self):
        from networks.tools import get_convert_matrix
        verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix(self.verb_num_classes, self.obj_num_classes)
        self.obj_to_HO_matrix = tf.constant(obj_to_HO_matrix, tf.float32)
        self.verb_to_HO_matrix = tf.constant(verb_to_HO_matrix, tf.float32)
        self.gt_obj_class = tf.cast(tf.matmul(self.gt_class_HO, self.obj_to_HO_matrix, transpose_b=True) > 0,
                                    tf.float32)
        self.gt_verb_class = tf.cast(tf.matmul(self.gt_class_HO, self.verb_to_HO_matrix, transpose_b=True) > 0,
                                     tf.float32)

        from networks.tools import get_word2vec
        word2vec = get_word2vec()
        self.word2vec_emb = tf.constant(word2vec)
        self.gt_class_HO_for_G_verbs = None
        self.gt_class_HO_for_D_verbs = None

    def build_base(self):
        with tf.variable_scope(self.scope, self.scope, reuse=tf.AUTO_REUSE,):
            net = resnet_utils.conv2d_same(self.image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net

    def image_to_head(self, is_training):
        print('image to head, ', cfg.RESNET.FIXED_BLOCKS)
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net    = self.build_base()
            net, _ = resnet_v1.resnet_v1(net,
                                         self.blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                         global_pool=False,
                                         include_root_block=False,
                                         reuse=tf.AUTO_REUSE,
                                         scope=self.scope)
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            stop = -2
            head, _ = resnet_v1.resnet_v1(net,
                                          self.blocks[cfg.RESNET.FIXED_BLOCKS:stop],
                                          global_pool=False,
                                          include_root_block=False,
                                          reuse=tf.AUTO_REUSE,
                                          scope=self.scope)
        return head

    def sp_to_head(self):
        with tf.variable_scope(self.scope, self.scope, reuse=tf.AUTO_REUSE,):
            ends = 2
            conv1_sp      = slim.conv2d(self.spatial[:,:,:,0:ends], 64, [5, 5], padding='VALID', scope='conv1_sp')
            pool1_sp      = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
            conv2_sp      = slim.conv2d(pool1_sp,     32, [5, 5], padding='VALID', scope='conv2_sp')
            pool2_sp      = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
            pool2_flat_sp = slim.flatten(pool2_sp)

        return pool2_flat_sp

    def res5(self, pool5_H, pool5_O, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            if pool5_H is None:
                fc7_H = None
            else:
                fc7_H, _ = resnet_v1.resnet_v1(pool5_H,
                                               self.blocks[-2:-1],
                                               global_pool=False,
                                               include_root_block=False,
                                               reuse=tf.AUTO_REUSE,
                                               scope=self.scope)

            if pool5_O is None:
                fc7_O = None
            else:
                fc7_O, _ = resnet_v1.resnet_v1(pool5_O,
                                           self.blocks[-1:],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=tf.AUTO_REUSE,
                                           scope=self.scope)


        return fc7_H, fc7_O

    def head_to_tail(self, fc7_H, fc7_O, pool5_SH, pool5_SO, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):

            fc7_SH = tf.reduce_mean(pool5_SH, axis=[1, 2])
            fc7_SO = tf.reduce_mean(pool5_SO, axis=[1, 2])

            Concat_SH     = tf.concat([fc7_H, fc7_SH], 1)
            fc8_SH        = slim.fully_connected(Concat_SH, self.num_fc, scope='fc8_SH', reuse=tf.AUTO_REUSE)
            fc8_SH        = slim.dropout(fc8_SH, keep_prob=0.5, is_training=is_training, scope='dropout8_SH')
            fc9_SH        = slim.fully_connected(fc8_SH, self.num_fc, scope='fc9_SH', reuse=tf.AUTO_REUSE)
            fc9_SH        = slim.dropout(fc9_SH,    keep_prob=0.5, is_training=is_training, scope='dropout9_SH')  

            Concat_SO     = tf.concat([fc7_O, fc7_SO], 1)
            fc8_SO        = slim.fully_connected(Concat_SO, self.num_fc, scope='fc8_SO', reuse=tf.AUTO_REUSE)
            fc8_SO        = slim.dropout(fc8_SO, keep_prob=0.5, is_training=is_training, scope='dropout8_SO')
            fc9_SO        = slim.fully_connected(fc8_SO, self.num_fc, scope='fc9_SO', reuse=tf.AUTO_REUSE)
            fc9_SO        = slim.dropout(fc9_SO,    keep_prob=0.5, is_training=is_training, scope='dropout9_SO')  

            Concat_SHsp   = tf.concat([fc7_H, sp], 1)
            Concat_SHsp   = slim.fully_connected(Concat_SHsp, self.num_fc, scope='Concat_SHsp', reuse=tf.AUTO_REUSE)
            Concat_SHsp   = slim.dropout(Concat_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout6_SHsp')
            fc7_SHsp      = slim.fully_connected(Concat_SHsp, self.num_fc, scope='fc7_SHsp', reuse=tf.AUTO_REUSE)
            fc7_SHsp      = slim.dropout(fc7_SHsp,  keep_prob=0.5, is_training=is_training, scope='dropout7_SHsp')

        return fc9_SH, fc9_SO, fc7_SHsp
        

    def crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:

            batch_ids    = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            bboxes = self.trans_boxes_by_feats(bottom, rois)
            if cfg.RESNET.MAX_POOL:
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
                crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
            else:
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE], name="crops")
        return crops

    def trans_boxes_by_feats(self, bottom, rois):
        bottom_shape = tf.shape(bottom)
        height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.stride[0])
        width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.stride[0])
        x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
        y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
        x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
        y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
        bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
        return bboxes

    def bottleneck(self, bottom, is_training, name, reuse=False):
        with tf.variable_scope(name) as scope:

            if reuse:
                scope.reuse_variables()

            head_bottleneck = slim.conv2d(bottom, 1024, [1, 1], scope=name)

        return head_bottleneck


    def create_architecture(self, is_training):

        self.build_network(is_training)

        # for var in tf.trainable_variables():
        #     self.train_summaries.append(var)

        if is_training: self.add_loss()
        layers_to_output = {}
        layers_to_output.update(self.losses)

        val_summaries = []
        if is_training:
            with tf.device("/cpu:0"):

                for key, var in self.visualize.items():
                    tf.summary.image(key, var, max_outputs=1)
                for key, var in self.event_summaries.items():
                    val_summaries.append(tf.summary.scalar(key, var))

                self.summary_op     = tf.summary.merge_all()
                self.summary_op_val = tf.summary.merge(val_summaries)

        return layers_to_output

    def add_score_summary(self, key, tensor):
        if tensor is not None and tensor.op is not None:
            tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)


    def add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def get_feed_dict(self, blobs):
        feed_dict = {self.image: blobs['image'], self.H_boxes: blobs['H_boxes'],
                     self.O_boxes: blobs['O_boxes'], self.gt_class_HO: blobs['gt_class_HO'],
                     self.spatial: blobs['sp'],
                     # self.lr: lr,
                     self.H_num: blobs['H_num']}
        return feed_dict


    def train_step_tfr(self, sess, blobs, lr, train_op):
        loss, image_id, _ = sess.run([self.losses['total_loss'], self.image_id,
                            train_op])
        return loss, image_id

    def train_step_tfr_with_summary(self, sess, blobs, lr, train_op):

        loss, summary, image_id,  _ = sess.run([self.losses['total_loss'],
                                     self.summary_op, self.image_id,
                                     train_op])
        return loss, image_id, summary
    
    def test_image_HO(self, sess, image, blobs):
        feed_dict = {self.image: image, self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'], self.spatial: blobs['sp'], self.H_num: blobs['H_num']}
        cls_prob_HO = sess.run([self.predictions["cls_prob_HO"]], feed_dict=feed_dict)

        return cls_prob_HO


    def init_verbs_objs_cls(self):
        pass

    def set_obj_class(self, obj_class):
        self.gt_obj_class = obj_class

    def res5_ho(self, pool5_HO, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            fc7_HO, _ = resnet_v1.resnet_v1(pool5_HO,
                                            self.blocks[-2:-1],
                                            global_pool=False,
                                            include_root_block=False,
                                            reuse=tf.AUTO_REUSE,
                                            scope=self.scope)

        return fc7_HO

    def head_to_tail_hoi(self, fc7_O, fc7_verbs, fc7_O_raw, fc7_verbs_raw, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):

            print('others concat')
            concat_verbs = tf.concat([fc7_verbs, fc7_O], 1)  # TODO fix
            print(concat_verbs)
            # concat_verbs = fc7_verbs
            concat_verbs = slim.fully_connected(concat_verbs, self.num_fc, reuse=tf.AUTO_REUSE,
                                                scope='Concat_verbs')
            concat_verbs = slim.dropout(concat_verbs, keep_prob=0.5, is_training=is_training,
                                        scope='dropout6_verbs')
            fc_hoi = slim.fully_connected(concat_verbs, self.num_fc, reuse=tf.AUTO_REUSE,
                                             scope='fc7_verbs')
            fc_hoi = slim.dropout(fc_hoi, keep_prob=0.5, is_training=is_training,
                                     scope='dropout7_verbs')

        return fc_hoi

    def head_to_tail_sp(self, fc7_H, fc7_O, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            Concat_SHsp   = tf.concat([fc7_H, sp], 1)
            Concat_SHsp   = slim.fully_connected(Concat_SHsp, self.num_fc, reuse=tf.AUTO_REUSE, scope='Concat_SHsp')
            Concat_SHsp   = slim.dropout(Concat_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout6_SHsp')
            fc7_SHsp      = slim.fully_connected(Concat_SHsp, self.num_fc, reuse=tf.AUTO_REUSE, scope='fc7_SHsp')
            fc7_SHsp      = slim.dropout(fc7_SHsp,  keep_prob=0.5, is_training=is_training, scope='dropout7_SHsp')

        return fc7_SHsp

    def region_classification_sp(self, fc7_SHsp, is_training, initializer, name):
        with tf.variable_scope(name) as scope:

            cls_score_sp = slim.fully_connected(fc7_SHsp, self.num_classes,
                                                weights_initializer=initializer,
                                                trainable=is_training,
                                                reuse=tf.AUTO_REUSE,
                                                activation_fn=None, scope='cls_score_sp')
            cls_prob_sp = tf.nn.sigmoid(cls_score_sp, name='cls_prob_sp')
            tf.reshape(cls_prob_sp, [-1, self.num_classes])

            self.predictions["cls_score_sp"] = cls_score_sp
            self.predictions["cls_prob_sp"] = cls_prob_sp

        return cls_prob_sp

    def region_classification_hoi(self, fc7_verbs, is_training, initializer, name, nameprefix =''):
        with tf.variable_scope(name) as scope:
            cls_score_verbs = slim.fully_connected(fc7_verbs, self.num_classes,
                                                       weights_initializer=initializer,
                                                       trainable=is_training,
                                                       reuse=tf.AUTO_REUSE,
                                                       activation_fn=None, scope='cls_score_verbs')
            cls_prob_verbs = tf.nn.sigmoid(cls_score_verbs, name='cls_prob_verbs')
            self.predictions[nameprefix+"cls_score_verbs"] = cls_score_verbs
            self.predictions[nameprefix+"cls_prob_verbs"] = cls_prob_verbs

            return cls_prob_verbs

    def get_compose_boxes(self, h_boxes, o_boxes):
        with tf.control_dependencies([tf.assert_equal(h_boxes[:, 0], o_boxes[:, 0],
                                                                data=[h_boxes[:, 0], o_boxes[:, 0]])]):
            cboxes1 = tf.minimum(tf.slice(h_boxes, [0, 0], [-1, 3]),
                                 tf.slice(o_boxes, [0, 0], [-1, 3]))
            cboxes2 = tf.maximum(tf.slice(h_boxes, [0, 3], [-1, 2]),
                                 tf.slice(o_boxes, [0, 3], [-1, 2]))
            cboxes = tf.concat(values=[cboxes1, cboxes2], axis=1)
            return cboxes


    def verbs_loss(self, fc7_verbs, is_training, initializer, label='',):
        with tf.variable_scope('verbs_loss', reuse=tf.AUTO_REUSE):
            cls_verbs = fc7_verbs
            verbs_cls_score = slim.fully_connected(cls_verbs, self.verb_num_classes,
                                                   weights_initializer=initializer,
                                                   trainable=is_training,
                                                   reuse=tf.AUTO_REUSE,
                                                   activation_fn=None, scope='verbs_cls_score')
            verb_cls_prob = tf.nn.sigmoid(verbs_cls_score, name='verb_cls_prob')
            tf.reshape(verb_cls_prob, [-1, self.verb_num_classes])

            self.predictions["verb_cls_score"+label] = verbs_cls_score
            self.predictions["verb_cls_prob"+label] = verb_cls_prob

    def objects_loss(self, input_feature, is_training, initializer, name = 'objects_loss', label='', is_stop_grads=False):
        """
        This is useless.
        """
        with tf.variable_scope(name):
            print('objects_loss:', self.model_name)
            obj_cls_score = slim.fully_connected(input_feature, self.obj_num_classes,
                                                   weights_initializer=initializer,
                                                   trainable=is_training,
                                                 reuse=tf.AUTO_REUSE,
                                                   activation_fn=None, scope='obj_cls_score')
            obj_cls_prob = tf.nn.sigmoid(obj_cls_score, name='obj_cls_prob')
            tf.reshape(obj_cls_prob, [-1, self.obj_num_classes])

            self.predictions["obj_cls_score"+label] = obj_cls_score
            self.predictions["obj_cls_prob"+label] = obj_cls_prob

    def build_network(self, is_training):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        num_stop = tf.cast(self.get_num_stop(), tf.int32)
        # ResNet Backbone
        head = self.image_to_head(is_training)
        sp = self.sp_to_head()
        print('sp======', sp)
        cboxes = self.get_compose_boxes(self.H_boxes, self.O_boxes)

        pool5_O = self.crop_pool_layer(head, self.O_boxes, 'Crop_O')
        pool5_H = self.crop_pool_layer(head, self.H_boxes, 'Crop_H')
        cboxes = cboxes[:num_stop]


        pool5_HO = self.extract_pool5_HO(head, cboxes, is_training, pool5_O, None, name='ho_')

        fc7_H_raw, fc7_O_raw = self.res5(pool5_H, pool5_O, None, is_training, 'res5')
        fc7_H = tf.reduce_mean(fc7_H_raw, axis=[1, 2])
        fc7_O = tf.reduce_mean(fc7_O_raw, axis=[1, 2])
        fc7_H_pos = fc7_H[:num_stop]
        fc7_O_pos = fc7_O[:num_stop]
        fc7_HO_raw = self.res5_ho(pool5_HO, is_training, 'res5')

        fc7_HO = None if fc7_HO_raw is None else tf.reduce_mean(fc7_HO_raw, axis=[1, 2])

        fc7_verbs_raw = fc7_HO_raw
        fc7_verbs = fc7_HO

        self.score_summaries.update({'orth_HO': fc7_HO,
                                     'orth_H': fc7_H, 'orth_O': fc7_O})

        print('sp', sp)
        fc7_SHsp = self.head_to_tail_sp(fc7_H, fc7_O, sp, is_training, 'fc_HO')
        cls_prob_sp = self.region_classification_sp(fc7_SHsp, is_training, initializer, 'classification')
        print("sp:", fc7_SHsp)

        self.verbs_loss(fc7_verbs, is_training, initializer)
        if self.model_name.__contains__('_objloss'):
            # This is useless
            self.objects_loss(fc7_O, is_training, initializer, 'objects_loss', label='_o')

        print('verbs')
        all_fc7_O = None
        if is_training:
            gt_class = self.gt_class_HO[:num_stop]
            tmp_fc7_O = fc7_O[:num_stop]
            tmp_fc7_verbs = fc7_verbs[:num_stop]
            tmp_O_raw = fc7_O_raw[:num_stop]
            fc7_O, fc7_verbs = self.feature_gen.fabricate_model(tmp_fc7_O, tmp_O_raw,
                                                                tmp_fc7_verbs, fc7_verbs_raw[:num_stop], initializer, is_training,
                                                                gt_class)

            # if self.model_name.__contains__('laobj'):
            #     # this aims to evaluate the effect of regularizing fabricated object features, we do not use.
            #     all_fc7_O = fc7_O
            #     tmp_class = self.gt_class_HO_for_D_verbs
            #     self.gt_obj_class = tf.cast(
            #         tf.matmul(tmp_class, self.obj_to_HO_matrix, transpose_b=True) > 0,
            #         tf.float32)
            #     self.objects_loss(all_fc7_O, is_training, initializer, 'objects_loss', label='_o')
            #     pass
        fc7_vo = self.head_to_tail_hoi(fc7_O, fc7_verbs, fc7_O_raw, fc7_verbs_raw, is_training, 'fc_HO')
        cls_prob_verbs = self.region_classification_hoi(fc7_vo, is_training, initializer, 'classification')
        if self.gt_class_HO_for_D_verbs is None:
            self.gt_class_HO_for_D_verbs = self.gt_class_HO[:num_stop]

        if self.model_name.__contains__('_l0_') or self.model_name.__contains__('_scale_'):
            """
            This is for factorized model.
            """
            verb_prob = self.predictions['verb_cls_prob']
            obj_prob = self.predictions["obj_cls_prob_o"]
            print(verb_prob, obj_prob)
            tmp_fc7_O_vectors = tf.cast(
                tf.matmul(obj_prob, self.obj_to_HO_matrix) > 0,
                tf.float32)
            tmp_fc7_verbs_vectors = tf.cast(
                tf.matmul(verb_prob, self.verb_to_HO_matrix) > 0,
                tf.float32)
            if 'cls_prob_verbs' not in self.predictions:
                self.predictions['cls_prob_verbs'] = 0
            if self.model_name.__contains__('_l0_'):
                self.predictions['cls_prob_verbs'] = 0
            self.predictions['cls_prob_verbs'] += (tmp_fc7_O_vectors + tmp_fc7_verbs_vectors)

        self.score_summaries.update(self.predictions)

    def get_num_stop(self):
        num_stop = tf.shape(self.H_boxes)[0]  # for selecting the positive items
        if self.model_name.__contains__('_new'):
            print('new Add H_num constrains')
            # This is the default setting.
            num_stop = self.H_num
        else:  # contain some negative items
            # this does not improve the baseline much.
            H_num_tmp = tf.cast(self.H_num, tf.int32)
            num_stop = tf.cast(num_stop, tf.int32)
            num_stop = H_num_tmp + tf.cast((num_stop - H_num_tmp) // 8, tf.int32)

        return num_stop

    def get_compose_num_stop(self):
        num_stop = self.get_num_stop()
        return num_stop

    def extract_pool5_HO(self, head, cboxes, is_training, pool5_O, head_mask = None, name=''):
        #
        if self.model_name.__contains__('_union'):
            pool5_HO = self.crop_pool_layer(head, cboxes, name + 'Crop_HO')
            self.test_visualize["pool5_HO"] = tf.expand_dims(tf.reduce_mean(pool5_HO, axis=-1), axis=-1)
        elif self.model_name.__contains__('_humans'):
            print("humans")
            pool5_HO = self.crop_pool_layer(head, self.H_boxes[:self.get_num_stop()],name +  'Crop_HO_h')
            self.test_visualize["pool5_HO"] = tf.expand_dims(tf.reduce_mean(pool5_HO, axis=-1), axis=-1)
        else:
            # pool5_HO = self.crop_pool_layer(head, cboxes, 'Crop_HO')
            pool5_HO = None
            print("{} doesn\'t support pool5_HO".format(self.model_name))
        return pool5_HO


    def add_loss_key_value(self, key, value):
        if key in self.losses:
            self.losses[key] += value
        else:
            self.losses[key] = value

    def add_loss(self):
        with tf.variable_scope('LOSS') as scope:

            num_stop = self.get_num_stop()

            label_H = self.gt_class_HO[:num_stop]
            # label_HO = self.gt_class_HO_for_verbs
            label_HO = self.gt_class_HO[:num_stop]
            label_sp = self.gt_class_HO
            if "cls_score_sp" in self.predictions:
                cls_score_sp = self.predictions["cls_score_sp"]
                cls_score_sp = tf.multiply(cls_score_sp, self.HO_weight)
                sp_cross_entropy = self.cal_sigmoid_entropy_loss(cls_score_sp, label_sp)

                self.losses['sp_cross_entropy'] = sp_cross_entropy


            tmp_label_HO = self.gt_class_HO_for_D_verbs
            cls_score_verbs = self.predictions["cls_score_verbs"][:tf.shape(self.gt_class_HO_for_D_verbs)[0], :]
            print('debug gt_class_HO_for_D_verbs:', self.gt_class_HO_for_D_verbs, cls_score_verbs)
            # reweight is from TIN
            cls_score_verbs = tf.multiply(cls_score_verbs, self.HO_weight)
            print('=======', tmp_label_HO, cls_score_verbs)

            tmp_verb_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tmp_label_HO, logits=cls_score_verbs)
            verbs_cross_entropy = tf.reduce_mean(tmp_verb_loss)
            self.losses['verbs_cross_entropy'] = verbs_cross_entropy

            lamb = 2
            if "cls_score_sp" not in self.predictions:
                sp_cross_entropy = 0
                self.losses['sp_cross_entropy'] = 0
            loss = sp_cross_entropy + verbs_cross_entropy * lamb

            if 'fake_G_cls_score_verbs' in self.predictions:
                fake_cls_score_verbs = self.predictions["fake_G_cls_score_verbs"]
                if self.model_name.__contains__('_rew_'):
                    fake_cls_score_verbs = tf.multiply(fake_cls_score_verbs, self.HO_weight)
                self.losses['fake_G_verbs_cross_entropy'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.gt_class_HO_for_G_verbs, logits=fake_cls_score_verbs))
                if 'fake_G_total_loss' not in self.losses:
                    self.losses['fake_G_total_loss'] = 0
                self.losses['fake_G_total_loss'] += (self.losses['fake_G_verbs_cross_entropy'] * 1)
            # verb loss
            temp = self.add_verb_loss(num_stop)
            loss += temp

            self.losses['total_loss'] = loss
            self.event_summaries.update(self.losses)
        print(self.losses)
        print(self.predictions)
        return loss

    def cal_sigmoid_entropy_loss(self, cls_score_sp, label_sp):
        sp_cross_entropy = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=label_sp, logits=cls_score_sp))
        return sp_cross_entropy

    def add_objloss(self, num_stop):
        obj_cls_score = self.predictions["obj_cls_score_o"]

        label_obj = tf.cast(tf.matmul(self.gt_class_HO_for_D_verbs, self.obj_to_HO_matrix, transpose_b=True) > 0,
                tf.float32)
        obj_cls_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label_obj[:tf.shape(obj_cls_score)[0], :], logits=obj_cls_score))
        self.losses["obj_cls_cross_entropy_o"] = obj_cls_cross_entropy

        model_name = self.model_name
        lambda1 = 0.3
        temp = (obj_cls_cross_entropy * lambda1)

        return temp

    def add_verb_loss(self, num_stop):
        temp = 0
        if 'verb_cls_score' in self.predictions:
            vloss_num_stop = num_stop
            verb_cls_score = self.predictions["verb_cls_score"]
            verb_cls_cross_entropy = self.cal_sigmoid_entropy_loss(verb_cls_score[:vloss_num_stop, :], self.gt_verb_class[:vloss_num_stop])
            self.losses["verb_cls_cross_entropy"] = verb_cls_cross_entropy
            # neg 0.1, negv1 0.5 negv12 0.1 1
            lambda1 = 0.3
            # lambda1 = 1.0
            # lambda1 = 2.0
            # lambda1 = 0.5
            # lambda1 = 0.1
            # lambda1 = 0.05
            temp = (verb_cls_cross_entropy * lambda1)
        return temp

    def train_step(self, sess, blobs, lr, train_op):
        feed_dict = self.get_feed_dict(blobs)

        loss, _ = sess.run([self.losses['total_loss'],
                            train_op],
                           feed_dict=feed_dict)
        return loss

    def train_step_with_summary(self, sess, blobs, lr, train_op):
        feed_dict = self.get_feed_dict(blobs)

        loss, summary, _ = sess.run([self.losses['total_loss'],
                                     self.summary_op,
                                     train_op],
                                    feed_dict=feed_dict)
        return loss, summary
