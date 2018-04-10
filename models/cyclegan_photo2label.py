#!/usr/bin/env python3
'''
CycleGAN for photo2label
'''
import tensorflow as tf                 # tf.__version__ : 1.4
import tensorflow.contrib.slim as slim
import numpy as np
import scipy.misc
import os, time
from collections import defaultdict, namedtuple
from progress.bar import IncrementalBar
from utils.tools import *
from utils.metrics import ssim, fid  # calculate structral simility between two images

from utils.image_reader import ImageReader
from utils import sgtools
from utils.sgtools import IMG_MEAN

from layers import ResidualBlock, SPPLayers, Mask
from layers.pspnet import PSPNet101, PSPNet50
from layers.cyclegan import generator_resnet as CycleG_resnet
from layers.cyclegan import discriminator as CycleD

time_stamp = time.strftime("%Y%m%d-%H%M")

# ---[ Session Configures
GpuConfig = tf.ConfigProto()
GpuConfig.gpu_options.allow_growth=True
#GpuConfig.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

class CYCLEGAN_PHOTO2LABEL:
    def __init__(self, 
            train_dataset               = 'datasets/cityscape/:list.txt',
            val_dataset                 = ['datasets/cityscape/:list_val.txt'],
            bs                          = 12,
            crop_size                   = '713,713',
            resize                      = '256,256',
            random_scale                = True,
            num_labels                  = 19,
            print_epoch                 = 100,
            max_epoch                   = 200000,
            g_lr                        = 0.0002,
            d_lr                        = 0.0002,
            g_epoch                     = 1,
            d_epoch                     = 1,
            #optimizer                   = {'eval': 'AdamOptimizer', 'args':{'beta1':0., 'beta2':0.9}},
            lm                          = 10,
            gpus                        = 4,
            loss_mode                   = 'sigmoid_cross_entropy',  #['lsgan', 'sce'/'basic']
            num_threads                 = 32,
            result_dir                  = 'results',
            summary_dir                 = 'summary',
            suffix                      = 'r2.1.5-sce_loss.tr_and_val-GLOSSuseDecodedLabelSegRecover', 
            name                        = 'CycleGAN-photo2label',
            **kwargs):

        self.pretrain_D_epoch           = -1

        train_dataset = train_dataset.split(':')
        val_dataset   = [dataset.split(':') for dataset in val_dataset]

        self.source_data                = dict(data_dir=train_dataset[0], data_list=os.path.join(*train_dataset))
        self.target_data                = [dict(data_dir=data[0], data_list=os.path.join(*data)) for data in val_dataset]
        self.batch_size                 = bs
        self.num_labels                 = num_labels
        self.crop_size                  = [int(n) for n in crop_size.split(',')]
        self.resize                     = [int(n) for n in resize.split(',')]
        self.random_scale               = random_scale
        self.print_epoch                = print_epoch
        self.max_epoch                  = max_epoch
        self.g_lr                       = float('%.0e' % g_lr)
        self.d_lr                       = float('%.0e' % d_lr)
        self.g_epoch                    = g_epoch
        self.d_epoch                    = d_epoch
        #self.optimizer                  = optimizer
        self.L1_lambda                  = lm
        self.gpus                       = gpus
        self.loss_mode                  = loss_mode
        self.result_dir                 = result_dir
        self.summary_dir                = summary_dir
        self.name                       = name
        self.suffix                     = suffix

        name = name.lower() + '.{}Gep_{}Dep.{:.0e}Glr_{:.0e}Dlr.{}bs'.format(g_epoch, d_epoch, g_lr, d_lr, bs)
        folder_name = name + '.' + time_stamp

        self.result_dir  = os.path.join('results', folder_name, suffix)
        self.ckpt_dir    = os.path.join(self.result_dir, 'checkpoints')
        self.summary_dir = os.path.join(summary_dir, folder_name + '.')
        self.name        = name + suffix
        dirs = [
                self.result_dir,
                self.ckpt_dir,
                self.summary_dir,
                ]
        for d in dirs:
            try:
                os.makedirs(d)
                print (colored('++ [Create]', 'green'), d)
            except FileExistsError:
                print (colored('! [Use]', 'red'), d)
            except ImportError:
                print ('[DIR]', d)

    @staticmethod
    def resnetG(inputs,
                first_hidden_dim=64, 
                output_channel=3, 
                reuse=False, 
                phase_train=False, 
                scope='resnetG'
                ):
        outputs_collections = scope+'_endpoints'
        OPTIONS = namedtuple('OPTIONS', 'gf_dim output_c_dim is_training') # same as `...('OPTIONS', ['gf_dim', 'output_c_dim', 'is_training'])`
        options = OPTIONS._make((first_hidden_dim, output_channel, phase_train))
        net, raw_output = CycleG_resnet(inputs, options, reuse, scope)
        outputs = {
                'tanh':     net,    #output: -1 ~ 1
                'raw':      raw_output,
                'softmax':  tf.nn.softmax(raw_output, dim=-1),
                'endpoints':tf.get_collection(outputs_collections)
                }
        return outputs['raw']


    @staticmethod
    def resnetD(inputs, 
                first_hidden_dim=64, 
                reuse=False, 
                phase_train=False, 
                scope='resnetD'
                ):
        outputs_collections = scope+'_endpoints'
        OPTIONS = namedtuple('OPTIONS', 'df_dim is_training') # same as `...('OPTIONS', ['gf_dim', 'output_c_dim', 'is_training'])`
        options = OPTIONS._make((first_hidden_dim, phase_train))
        net = CycleD(inputs, options, reuse, scope)
        outputs = {
                'critic':   net,
                'sigmoid':  tf.nn.sigmoid(net),
                'endpoints':tf.get_collection(outputs_collections)
                }
        return outputs['critic']



    def build(self):
        config = self.__dict__.copy()
        num_labels      = self.num_labels    #for segmentation (pixel labels)
        ignore_label    = 255   #for segmentation (pixel labels)
        random_seed     = 1234
        generator       = self.resnetG
        discriminator   = self.resnetD
        GEN_A2B_NAME = 'GEN_A2B'
        GEN_B2A_NAME = 'GEN_B2A'
        DIS_A_NAME   = 'DIS_A'
        DIS_B_NAME   = 'DIS_B'

        global_step = tf.train.get_or_create_global_step()
        slim.add_model_variable(global_step)
        global_step_update = tf.assign_add(global_step, 1, name='global_step_update')

        def resize_and_onehot(tensor, shape, depth):
            with tf.device('/device:CPU:0'):
                onehot_tensor = tf.one_hot(tf.squeeze( 
                                        tf.image.resize_nearest_neighbor(
                                            tf.cast(tensor, tf.int32), shape), -1), depth=depth)
                return onehot_tensor
        def convert_to_labels(onehot_seg, crop_size=None):
            fake_segments_output = onehot_seg
            print ('%s | ' % fake_segments_output.device, fake_segments_output)
            if crop_size:
                fake_segments_output = tf.image.resize_bilinear(fake_segments_output, crop_size) #tf.shape(source_segments_batch)[1:3])
            fake_segments_output = tf.argmax(fake_segments_output, axis=-1) # generate segment indices matrix
            fake_segments_output = tf.expand_dims(fake_segments_output, dim=-1) # Create 4-d tensor.
            return fake_segments_output

        target_data_queue = []
        tf.set_random_seed(random_seed)
        coord = tf.train.Coordinator()
        with tf.name_scope("create_inputs"):
            for i, data in enumerate([config['source_data']] + config['target_data']):
                reader = ImageReader(
                    data['data_dir'],
                    data['data_list'],
                    config['crop_size'],                    # Original size: [1024, 2048]
                    random_scale=config['random_scale'],
                    random_mirror=True,
                    ignore_label=ignore_label,
                    img_mean=0,                             # set IMG_MEAN to centralize image pixels (set NONE for automatic choosing)
                    img_channel_format='RGB',               # Default: BGR in deeplab_v2. See here: https://github.com/zhengyang-wang/Deeplab-v2--ResNet-101--Tensorflow/issues/30
                    coord=coord,
                    rgb_label=False)
                data_queue = reader.dequeue(config['batch_size'])

                if i == 0:
                    # ---[ source: training data
                    source_images_batch    = data_queue[0]  #A: 3 chaanels
                    source_segments_batch  = data_queue[1]  #B: 1-label channels

                    source_images_batch    = tf.cast(source_images_batch, tf.float32) / 127.5 - 1.

                    source_images_batch    = tf.image.resize_bilinear(source_images_batch, config['resize'])  #A: 3 chaanels
                    source_segments_batch  = tf.image.resize_nearest_neighbor(source_segments_batch, config['resize'])  #B: 1-label channels

                    source_segments_batch  = tf.cast(tf.one_hot(tf.squeeze(source_segments_batch, -1), depth=num_labels), tf.float32) - 0.5 #B: 19 channels

                else:
                    # ---[ target: validation data / testing data
                    target_images_batch    = data_queue[0]  #A: 3 chaanels
                    target_segments_batch  = data_queue[1]  #B: 1-label channels

                    target_images_batch    = tf.cast(target_images_batch, tf.float32) / 127.5 - 1.

                    target_images_batch    = tf.image.resize_bilinear(target_images_batch, config['resize'])  #A: 3 chaanels
                    target_segments_batch  = tf.image.resize_nearest_neighbor(target_segments_batch, config['resize'])  #B: 1-label channels

                    target_segments_batch  = tf.cast(tf.one_hot(tf.squeeze(target_segments_batch, -1), depth=num_labels), tf.float32) - 0.5 #B: 19 channels
                    target_data_queue.append([target_images_batch, target_segments_batch])


        size_list = cuttool(config['batch_size'], config['gpus'])
        source_images_batches    = tf.split(source_images_batch,   size_list)
        source_segments_batches  = tf.split(source_segments_batch, size_list)
        fake_1_segments_output   = [None] * len(size_list)
        fake_2_segments_output   = [None] * len(size_list)
        fake_1_images_output     = [None] * len(size_list)
        fake_2_images_output     = [None] * len(size_list)
        d_real_img_output        = [None] * len(size_list)
        d_fake_img_output        = [None] * len(size_list)
        d_real_seg_output        = [None] * len(size_list)
        d_fake_seg_output        = [None] * len(size_list)

        for gid, (source_images_batch, source_segments_batch) in \
                enumerate(zip(source_images_batches, source_segments_batches)):
            # ---[ Generator A2B & B2A
            with tf.device('/device:GPU:{}'.format((gid-1) % config['gpus'])):
                fake_seg  = generator(source_images_batch, output_channel=num_labels, reuse=tf.AUTO_REUSE, phase_train=True, scope=GEN_A2B_NAME)
                fake_seg  = tf.nn.softmax(fake_seg) - 0.5
                fake_img_ = generator(fake_seg, output_channel=3, reuse=tf.AUTO_REUSE, phase_train=True, scope=GEN_B2A_NAME)
                fake_img_ = tf.nn.tanh(fake_img_)
                fake_img  = generator(source_segments_batch, output_channel=3, reuse=tf.AUTO_REUSE, phase_train=True, scope=GEN_B2A_NAME)
                fake_img  = tf.nn.tanh(fake_img)
                fake_seg_ = generator(fake_img, output_channel=num_labels, reuse=tf.AUTO_REUSE, phase_train=True, scope=GEN_A2B_NAME)
                fake_seg_ = tf.nn.softmax(fake_seg_) - 0.5

            # ---[ Discriminator A & B
            with tf.device('/device:GPU:{}'.format((gid-1) % config['gpus'])):
                d_real_img = discriminator(source_images_batch,   reuse=tf.AUTO_REUSE, phase_train=True, scope=DIS_A_NAME)
                d_fake_img = discriminator(fake_img, reuse=tf.AUTO_REUSE, phase_train=True, scope=DIS_A_NAME)
                d_real_seg = discriminator(source_segments_batch, reuse=tf.AUTO_REUSE, phase_train=True, scope=DIS_B_NAME)
                d_fake_seg = discriminator(fake_seg, reuse=tf.AUTO_REUSE, phase_train=True, scope=DIS_B_NAME)
                #d_fake_img_val = discriminator(fake_img_val, reuse=tf.AUTO_REUSE, phase_train=False, scope=DIS_A_NAME)
                #d_fake_seg_val = discriminator(fake_seg_val, reuse=tf.AUTO_REUSE, phase_train=False, scope=DIS_B_NAME)


                fake_1_segments_output [gid]  = fake_seg
                fake_2_segments_output [gid]  = fake_seg_
                fake_1_images_output [gid]    = fake_img
                fake_2_images_output [gid]    = fake_img_

                d_real_img_output [gid]       = d_real_img
                d_fake_img_output [gid]       = d_fake_img
                d_real_seg_output [gid]       = d_real_seg
                d_fake_seg_output [gid]       = d_fake_seg

        source_images_batch    = tf.concat(source_images_batches, axis=0)   #-1~1
        source_segments_batch  = tf.concat(source_segments_batches, axis=0) #onehot: -0.5~+0.5
        fake_1_segments_output = tf.concat(fake_1_segments_output, axis=0)  ;   print('fake_1_segments_output', fake_1_segments_output)
        fake_2_segments_output = tf.concat(fake_2_segments_output, axis=0)  ;   print('fake_2_segments_output', fake_2_segments_output)
        fake_1_images_output   = tf.concat(fake_1_images_output  , axis=0)  ;   print('fake_1_images_output  ', fake_1_images_output  )
        fake_2_images_output   = tf.concat(fake_2_images_output  , axis=0)  ;   print('fake_2_images_output  ', fake_2_images_output  )
        d_real_img_output      = tf.concat(d_real_img_output , axis=0)
        d_fake_img_output      = tf.concat(d_fake_img_output , axis=0)
        d_real_seg_output      = tf.concat(d_real_seg_output , axis=0)
        d_fake_seg_output      = tf.concat(d_fake_seg_output , axis=0)

        source_data_color = [
            (1.+source_images_batch   ) / 2.                                                                ,         # source_images_batch_color
            sgtools.decode_labels(tf.cast(convert_to_labels(source_segments_batch + 0.5), tf.int32),  num_labels),    # source_segments_batch_colo
            sgtools.decode_labels(tf.cast(convert_to_labels(fake_1_segments_output + 0.5), tf.int32),  num_labels),   # fake_1_segments_output_col
            sgtools.decode_labels(tf.cast(convert_to_labels(fake_2_segments_output + 0.5), tf.int32),  num_labels),   # fake_2_segments_output_col
            (1.+fake_1_images_output  ) / 2.                                                                ,         # fake_1_images_output_color
            (1.+fake_2_images_output  ) / 2.                                                                ,         # fake_2_images_output_color
            ]

        # ---[ Validation Model
        target_data_color_queue = []
        for target_data in target_data_queue:
            with tf.device('/device:GPU:{}'.format((2) % config['gpus'])):
                fake_seg  = generator(val_images_holder, output_channel=num_labels, reuse=tf.AUTO_REUSE, phase_train=False, scope=GEN_A2B_NAME)
                fake_seg  = tf.nn.softmax(fake_seg) - 0.5
                fake_img_ = generator(fake_seg, output_channel=3, reuse=tf.AUTO_REUSE, phase_train=False, scope=GEN_B2A_NAME)
                fake_img_ = tf.nn.tanh(fake_img_)
                fake_img  = generator(val_segments_holder, output_channel=3, reuse=tf.AUTO_REUSE, phase_train=False, scope=GEN_B2A_NAME)
                fake_img  = tf.nn.tanh(fake_img)
                fake_seg_ = generator(fake_img, output_channel=num_labels, reuse=tf.AUTO_REUSE, phase_train=False, scope=GEN_A2B_NAME)
                fake_seg_ = tf.nn.softmax(fake_seg) - 0.5

            target_data_color_queue.append([
                    (1.+target_images_batch   ) / 2.                                                          , # target_images_batch_color
                    sgtools.decode_labels(tf.cast(convert_to_labels(target_segments_batch + 0.5), tf.int32),  num_labels)    , # target_segments_batch_color
                    sgtools.decode_labels(tf.cast(convert_to_labels(fake_seg  + 0.5), tf.int32),  num_labels) , # val_fake_1_segments_output_color
                    sgtools.decode_labels(tf.cast(convert_to_labels(fake_seg_ + 0.5), tf.int32),  num_labels) , # val_fake_2_segments_output_color
                    (1.+val_fake_1_images_output  ) / 2.                                                      , # val_fake_1_images_output_color
                    (1.+val_fake_2_images_output  ) / 2.                                                      , # val_fake_2_images_output_color
                    ])

        # ---[ Segment-level loss: pixelwise loss
        # d_seg_batch = tf.image.resize_nearest_neighbor(seg_gt, tf.shape(_d_real['segment'])[1:3])
        # d_seg_batch = tf.squeeze(d_seg_batch, -1)
        # d_seg_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=d_seg_batch, logits=_d_real['segment'], name='segment_pixelwise_loss')   # pixel-wise loss
        # d_seg_loss = tf.reduce_mean(d_seg_loss)
        # d_seg_loss = tf.identity(d_seg_loss, name='d_seg_loss')

        # ---[ GAN Loss: crite loss
        #d_loss_old = - (tf.reduce_mean(d_source_output['critic']) - tf.reduce_mean(d_target_output['critic']))
        #g_loss = - (tf.reduce_mean(d_target_output['critic']))
        ## gradient penalty
        #LAMBDA = 10
        ##alpha = tf.placeholder(tf.float32, shape=[None], name='alpha')
        #alpha = tf.random_uniform([config['batch_size']], 0.0, 1.0, dtype=tf.float32)
        #for _ in source_segments_batch.shape[1:]:
            #alpha = tf.expand_dims(alpha, axis=1)   #shape=[None,1,1,1]
        #interpolates = alpha * source_segments_batch + (1.-alpha) * target_segments_output
        #print ('source_segments_batch:', source_segments_batch)
        #print ('target_segments_output:',target_segments_output)
        #print ('interpolates:', interpolates)
        #interpolates = resize_and_onehot(interpolates, target_raw_segments_output.shape.as_list()[1:3], num_labels)
        #print ('interpolates:', interpolates)
        #_d_intp = discriminator(interpolates, reuse=True, phase_train=True, scope=DIS_NAME)
        #intp_grads = tf.gradients(_d_intp['critic'], [interpolates])[0]
        #slopes = tf.sqrt(tf.reduce_sum(tf.square(intp_grads), reduction_indices=[1]))   #L2-distance
        #grads_penalty = tf.reduce_mean(tf.square(slopes-1), name='grads_penalty')
        #d_loss = d_loss_old + LAMBDA * grads_penalty


        def sigmoid_cross_entropy(labels, logits):
            return tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits) )
        def least_square(labels, logits):
            return tf.reduce_mean( (labels - logits) ** 2 )

        if config['loss_mode'] == 'lsgan':
            # ---[ GAN loss: LSGAN loss (chi-square, or called least-square)
            loss_func = least_square
        else:
            # ---[ GAN loss: sigmoid BCE loss
            loss_func = sigmoid_cross_entropy

        # ---[ LOSS
        _img_recovery = config['L1_lambda'] * tf.reduce_mean( tf.abs(source_images_batch - fake_2_images_output))
        #_seg_recovery = config['L1_lambda'] * tf.reduce_mean( tf.abs(source_segments_batch - fake_1_segments_output))   #r1.0: error
        #_seg_recovery = config['L1_lambda'] * tf.reduce_mean( tf.abs(source_segments_batch - fake_2_segments_output))   #r2.0
        _seg_recovery = config['L1_lambda'] * tf.reduce_mean( tf.abs(source_segments_batch_color - fake_2_segments_output_color))    #r2.0.5: not sure because, in theory, no gradient if using decode_labels()


        g_loss_a2b = \
                loss_func( labels=tf.ones_like(d_fake_seg_output), logits=d_fake_seg_output ) + \
                _img_recovery + _seg_recovery
        g_loss_b2a = \
                loss_func( labels=tf.ones_like(d_fake_img_output), logits=d_fake_img_output ) + \
                _img_recovery + _seg_recovery
        g_loss = \
                loss_func( labels=tf.ones_like(d_fake_seg_output), logits=d_fake_seg_output ) + \
                loss_func( labels=tf.ones_like(d_fake_img_output), logits=d_fake_img_output ) + \
                _img_recovery + _seg_recovery

        da_loss = \
                loss_func( labels=tf.ones_like(d_real_img_output), logits=d_real_img_output ) + \
                loss_func( labels=tf.zeros_like(d_fake_img_output), logits=d_fake_img_output )
        db_loss = \
                loss_func( labels=tf.ones_like(d_real_seg_output), logits=d_real_seg_output ) + \
                loss_func( labels=tf.zeros_like(d_fake_seg_output), logits=d_fake_seg_output )
        d_loss = \
                (da_loss + db_loss) / 2.

        # D will output [BATCH_SIZE, 32, 32, 1]
        num_da_real_img_acc = tf.size( tf.where(tf.reduce_mean(tf.nn.sigmoid(d_real_img_output), axis=[1,2,3]) > 0.5)[:,0], name='num_da_real_img_acc' )
        num_da_fake_img_acc = tf.size( tf.where(tf.reduce_mean(tf.nn.sigmoid(d_fake_img_output), axis=[1,2,3]) < 0.5)[:,0], name='num_da_fake_img_acc' )
        num_db_real_seg_acc = tf.size( tf.where(tf.reduce_mean(tf.nn.sigmoid(d_real_seg_output), axis=[1,2,3]) > 0.5)[:,0], name='num_db_real_seg_acc' )
        num_db_fake_seg_acc = tf.size( tf.where(tf.reduce_mean(tf.nn.sigmoid(d_fake_seg_output), axis=[1,2,3]) < 0.5)[:,0], name='num_db_fake_seg_acc' )

        ## limit weights to 0
        #g_weight_regularizer = [0.0001 * tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GEN_NAME) if 'weight' in v.name]
        #g_weight_regularizer = tf.add_n(g_weight_regularizer, name='g_weight_regularizer_loss')
        #g_loss += g_weight_regularizer
        #d_weight_regularizer = [0.0001 * tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, DIS_NAME) if 'weight' in v.name]
        #d_weight_regularizer = tf.add_n(d_weight_regularizer, name='d_weight_regularizer_loss')
        #d_loss += d_weight_regularizer

        d_loss = tf.identity(d_loss, name='d_loss')
        g_loss = tf.identity(g_loss, name='g_loss')

        ## --- Training Set Validation ---
        # Predictions.
        #pred_gt = tf.reshape(target_segments_batch, [-1,])
        #pred    = tf.reshape(target_segments_output, [-1,])
        #indices = tf.squeeze(tf.where(tf.not_equal(pred_gt, ignore_label)), 1)
        #pred_gt = tf.cast(tf.gather(pred_gt, indices), tf.int32)
        #pred    = tf.cast(tf.gather(pred, indices), tf.int32)
        ## mIoU
        ### Allowing to use indices matrices in mean_iou() with `num_classes=indices.max()`
        #weights = tf.cast(tf.less_equal(pred_gt, num_labels), tf.int32) # Ignoring all labels greater than or equal to n_classes.
        #mIoU, mIoU_update_op = tf.metrics.mean_iou(pred, pred_gt, num_classes=num_labels, weights=weights)

        # ---[ Variables
        g_a2b_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GEN_A2B_NAME)
        g_b2a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GEN_B2A_NAME)
        d_a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, DIS_A_NAME)
        d_b_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, DIS_B_NAME)
        g_vars = g_a2b_vars + g_b2a_vars
        d_vars = d_a_vars + d_b_vars

        print_list(g_a2b_vars, GEN_A2B_NAME)
        print_list(g_b2a_vars, GEN_B2A_NAME)
        print_list(d_a_vars, DIS_A_NAME)
        print_list(d_b_vars, DIS_B_NAME)

        # ---[ Optimizer
        ## `colocate_gradients_with_ops = True` to reduce GPU MEM utils, and fasten training speed
        OPT_NAME = 'Optimizer'
        g_opts = []; d_opts = []
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.variable_scope(OPT_NAME):
                #with tf.device('/device:GPU:{}'.format(config['gpus']-1)):
                if True:
                    if len(g_vars) > 0:
                        g_opt = tf.train.AdamOptimizer(learning_rate=config['g_lr'], beta1=0.5, beta2=0.9).minimize(g_loss,
                            var_list=g_vars, colocate_gradients_with_ops=True)
                        g_opts.append(g_opt)
                    if len(d_vars) > 0:
                        d_opt = tf.train.AdamOptimizer(learning_rate=config['d_lr'], beta1=0.5, beta2=0.9).minimize(d_loss,
                            var_list=d_vars, colocate_gradients_with_ops=True)
                        d_opts.append(d_opt)

        g_opt = tf.group(*g_opts)
        d_opt = tf.group(*d_opts)
        opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, OPT_NAME)
        print_list(opt_vars, OPT_NAME)

        # --- [ Summary
        scalars   = [d_loss, g_loss]
        #scalars  += [mIoU]
        scalars  += [num_da_real_img_acc, num_da_fake_img_acc, num_db_real_seg_acc, num_db_fake_seg_acc]
        scalars  += [g_loss_a2b, g_loss_b2a, da_loss, db_loss]
        writer, summarys = create_summary(summary_dir=config['summary_dir'], name=config['suffix'],
                scalar = scalars,
                )

        '''
        Training
        '''
        with tf.Session(config=GpuConfig) as sess:
            sess.run(tf.global_variables_initializer()) #DONOT put it after ``saver.restore``
            sess.run(tf.local_variables_initializer()) #DONOT put it after ``saver.restore``
            saver = tf.train.Saver(g_vars + d_vars, max_to_keep=1)
            #g_saver = tf.train.Saver(g_vars, max_to_keep=1)
            #d_saver = tf.train.Saver(d_vars, max_to_keep=1)
            #if self.ckpt:
                #saver.restore(sess, self.ckpt)
                #print ("Training starts at %d iteration..." % sess.run(global_step))

            feeds = {}

            # Start queue threads.
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            inside_epoch  = int(config['print_epoch']) if config['print_epoch'] < config['max_epoch'] else int(config['max_epoch'] / 1)
            outside_epoch = int(config['max_epoch'] / inside_epoch)
            start = int(sess.run(global_step) / inside_epoch)
            if start >= outside_epoch:
                raise ValueError("initial iteration:%d >= max iteration:%d. please reset '--max_epoch' value." % (sess.run(global_step), config['max_epoch']))

            start_time = time.time()
            for epo in range(start, outside_epoch):
                bar = IncrementalBar('[epoch {:<4d}/{:<4d}]'.format(epo, outside_epoch), max=inside_epoch)
                for epi in range(inside_epoch):
                    iters = sess.run(global_step)
                    # save summary
                    if epo == 0:
                        save_summarys = sess.run(summarys, feed_dict=feeds)
                        writer.add_summary(save_summarys, iters)

                    for _ in range(config['d_epoch']):
                        sess.run(d_opt, feed_dict=feeds)

                    if iters > self.pretrain_D_epoch:
                        for _ in range(config['g_epoch']):
                            sess.run(g_opt, feed_dict=feeds)

                    sess.run(global_step_update)
                    bar.next()

                duration = time.time() - start_time
                disc_loss, gen_loss = \
                        sess.run([d_loss, g_loss], feed_dict=feeds)
                na_real, na_fake, nb_real, nb_fake = \
                        sess.run([num_da_real_img_acc, num_da_fake_img_acc, num_db_real_seg_acc, num_db_fake_seg_acc], feed_dict=feeds)

                #sess.run(mIoU_update_op, feed_dict=feeds)
                #miou = sess.run(mIoU, feed_dict=feeds)
                print (' -',
                        'DLoss: %-8.2e' % disc_loss,
                        #'(W: %-8.2e)' % disc_wloss,
                        'GLoss: %-8.2e' % gen_loss,
                        #'(W: %-8.2e)' % gen_wloss,
                        '|',
                        '[Da_img] #real: %d, #fake: %d' % (na_real, na_fake),
                        '[Db_seg] #real: %d, #fake: %d' % (nb_real, nb_fake),
                        '|',
                        #'[train_mIoU] %.2f' % miou,
                        '[ETA] %s' % format_time(duration)
                        )
                bar.finish()

                iters = sess.run(global_step)
                # save checkpoint
                if epo % 2 == 0:
                    saver_path = os.path.join(config['ckpt_dir'], '{}.ckpt'.format(config['name']))
                    saver.save(sess, save_path=saver_path, global_step=global_step)
                # save summary
                if epo % 1 == 0:
                    save_summarys = sess.run(summarys, feed_dict=feeds)
                    writer.add_summary(save_summarys, iters)
                # output samples
                if epo % 5 == 0:
                    img_gt, seg_gt, seg_1, seg_2, img_1, img_2 = sess.run(source_data_color)
                    print ("Range %10s:" % "seg_gt", seg_gt.min(), seg_gt.max())
                    print ("Range %10s:" % "seg_1", seg_1.min(), seg_1.max())
                    print ("Range %10s:" % "seg_2", seg_2.min(), seg_2.max())
                    print ("Range %10s:" % "img_gt", img_gt.min(), img_gt.max())
                    print ("Range %10s:" % "img_1", img_1.min(), img_1.max())
                    print ("Range %10s:" % "img_2", img_2.min(), img_2.max())
                    _output = np.concatenate([img_gt, seg_gt, seg_1, img_1, img_2, seg_2], axis=0)
                    save_visualization(_output, save_path=os.path.join(config['result_dir'], 'tr-{}.jpg'.format(iters)), size=[3, 2*config['batch_size']])
                    #seg_output = np.concatenate([seg_gt, seg_2, seg_1], axis=0)
                    #img_output = np.concatenate([img_gt, img_2, img_1], axis=0)
                    #save_visualization(seg_output, save_path=os.path.join(config['result_dir'], 'tr-seg-1gt_2mapback_3map-{}.jpg'.format(iters)), size=[3, config['batch_size']])
                    #save_visualization(img_output, save_path=os.path.join(config['result_dir'], 'tr-img-1gt_2mapback_3map-{}.jpg'.format(iters)), size=[3, config['batch_size']])
                    for i,target_data_color in enumerate(target_data_color_queue):
                        val_img_gt, val_seg_gt, val_seg_1, val_seg_2, val_img_1, val_img_2 = sess.run(target_data_color)
                        print ("Val Range %10s:" % "seg_gt", val_seg_gt.min(), val_seg_gt.max())
                        print ("Val Range %10s:" % "seg_1", val_seg_1.min(), val_seg_1.max())
                        print ("Val Range %10s:" % "seg_2", val_seg_2.min(), val_seg_2.max())
                        print ("Val Range %10s:" % "img_gt", val_img_gt.min(), val_img_gt.max())
                        print ("Val Range %10s:" % "img_1", val_img_1.min(), val_img_1.max())
                        print ("Val Range %10s:" % "img_2", val_img_2.min(), val_img_2.max())
                        _output = np.concatenate([val_img_gt, val_seg_gt, val_seg_1, val_img_1, val_img_2, val_seg_2], axis=0)
                        save_visualization(_output, save_path=os.path.join(config['result_dir'], 'val{}-{}.jpg'.format(i,iters)), size=[3, 2*config['batch_size']])
                        #val_seg_output = np.concatenate([val_seg_gt, val_seg_2, val_seg_1], axis=0)
                        #val_img_output = np.concatenate([val_img_gt, val_img_2, val_img_1], axis=0)
                        #save_visualization(seg_output, save_path=os.path.join(config['result_dir'], 'val{}-seg-1gt_2mapback_3map-{}.jpg'.format(i,iters)), size=[3, config['batch_size']])
                        #save_visualization(img_output, save_path=os.path.join(config['result_dir'], 'val{}-img-1gt_2mapback_3map-{}.jpg'.format(i,iters)), size=[3, config['batch_size']])

                writer.flush()
            writer.close()

# ---------------------------------------------------------------------------------------------------------------------------------------------
def create_summary(summary_dir, name, **kwargs):
    writer = tf.summary.FileWriter(os.path.join(summary_dir, name))
    summarys = []
    for key,value in kwargs.items():
        if key in 'histogram':
            summarys += [tf.summary.histogram('hist-' + item.op.name, item) for item in value]
            print_list (value, key)
        elif key in 'scalar':
            summarys += [tf.summary.scalar('s-' + item.op.name, item) for item in value]
            print_list (value, key)
        else:
            summarys += [tf.summary.histogram(str(key) + '-' + item.op.name, item) for item in value]
            print_list (value, key)
    return writer, tf.summary.merge(summarys)

def save_visualization(objs, save_path='./train_sample/sample.jpg', size=None):
    if size is None:
        size = (int(scipy.ceil(scipy.sqrt(len(objs)))), int(scipy.ceil(scipy.sqrt(len(objs)))))
    h, w = objs.shape[1], objs.shape[2]
    if len(objs.shape) == 3 or objs.shape[-1] == 1:  #no channel
        img = np.zeros((h * size[0], w * size[1]))
        if objs.shape[-1] == 1:
            objs = np.squeeze(objs, axis=-1)
    else:
        img = np.zeros((h * size[0], w * size[1], 3))
    for n, obj in enumerate(objs):
        row = int(n / size[1])
        col = int(n % size[1])
        img[row*h:(row+1)*h, col*w:(col+1)*w, ...] = obj
    scipy.misc.imsave(save_path, img)
    print (colored(' [SAVE]', 'green'), save_path)

if __name__ == '__main__':
    model = CYCLEGAN_PHOTO2LABEL()
    model.build()
