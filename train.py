import os
import multiprocessing
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils import optimizer, gradproc
import pdb
import numpy  as np

try:
    from .cfgs.config import cfg
    from .reader import Data
except Exception:
    from cfgs.config import cfg
    from reader import Data

# if cfg.backbone == 'vgg19':
#     try:
#         from .modules import VGGBlock as Backbone, Stage1Block, StageTBlock
#     except Exception:
#         from modules import VGGBlock as Backbone, Stage1Block, StageTBlock
# else:
#     try:
#         # from .modules import Mobilenetv2Block as Backbone, Stage1DepthBlock as Stage1Block, StageTDepthBlock as StageTBlock
#         from .modules import Mobilenetv2Block as Backbone, Stage1Block, StageTBlock
#     except Exception:
#         # from modules import Mobilenetv2Block as Backbone, Stage1DepthBlock as Stage1Block, StageTDepthBlock as StageTBlock
#         from modules import Mobilenetv2Block as Backbone, Stage1Block, StageTBlock

# def apply_mask(t, mask):
#     return t * mask

def image_preprocess(image, bgr=True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        image = image * (1.0 / 255)

        mean = [0.485, 0.456, 0.406]    # rgb
        std = [0.229, 0.224, 0.225]
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_std
        return image

class Model(ModelDesc):

    def __init__(self):
        pass

    def _get_inputs(self):
        return [
            InputDesc(tf.float32, (None, 256, 192, 3), 'imgs'),
            InputDesc(tf.float32, (None, cfg.final_num_joints, 64, 48), 'target'),
            InputDesc(tf.float32, (None, cfg.final_num_joints, 1), 'target_weight')
            # InputDesc(tf.float32, (None, None, None, 1), 'meta')
        ]

    def _build_graph(self, inputs):
        imgs, target, target_weight = inputs

        l = image_preprocess(imgs, bgr=True)

        # l = tf.cast(imgs, tf.float32) / 255.0 - 1


        #########################
        # ResNets
        #########################
        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                return Conv2D('convshortcut', l, n_out, 1, stride=stride)
            else:
                return l

        def basicblock(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3)
            return l + shortcut(input, ch_in, ch_out, stride)

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[1]
            if preact == 'both_preact':
                l = BNReLU('preact', l)
                input = l
            elif preact != 'no_preact':
                input = l
                l = BNReLU('preact', l)
            else:
                input = l
            l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            return l + shortcut(input, ch_in, ch_out * 4, stride)

        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'default')
                return l

        net_cfg = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck)
        }
        defs, block_func = net_cfg[cfg.depth]
        
        def get_deconv_cfg(deconv_kernel, index):
            if deconv_kernel == 4 or deconv_kernel == 3:
                padding="SAME"
            else:
                padding="VALID"
            return deconv_kernel, padding

        def make_deconv_layer(l, layernanme, num_layers, num_filters, num_kernels):
            assert num_layers == len(num_filters), \
                'ERROR: num_deconv_layers is different len(num_deconv_filters)'
            assert num_layers == len(num_kernels), \
                'ERROR: num_deconv_layers is different len(num_deconv_filters)'

            with tf.variable_scope(layernanme):
                for j in range(num_layers):
                    with tf.variable_scope('block{}'.format(j)):

                        kernel, padding = get_deconv_cfg(num_kernels[j], j)
                        planes = num_filters[j]
                        l = Deconv2D('deconv', l, filters=planes, kernel_size=kernel, strides=2, padding=padding, \
                        use_bias=cfg.deconv_with_bias, nl=BNReLU)

                return l

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NHWC'):
                logits = (LinearWrap(l)
                          .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                          .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                          .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                          .apply(layer, 'group1', block_func, 128, defs[1], 2)
                          .apply(layer, 'group2', block_func, 256, defs[2], 2)
                          .apply(layer, 'group3', block_func, 512, defs[3], 2)
                          .apply(make_deconv_layer, 'deconv_layers', cfg.num_deconv_layers, cfg.num_deconv_filters, cfg.num_deconv_kernels)
                          .Conv2D('final_layer', cfg.final_num_joints, cfg.final_conv_kernel, padding='SAME' if cfg.final_conv_kernel == 3 else 'VALID')())
        
        # pdb.set_trace()
        output1 = tf.identity(logits,  name = 'heatmaps')

        logits = tf.transpose(logits, [0,3,1,2])#n c h w

        heatmaps_ = tf.reshape(logits, [-1, 17, 64*48])
        heatmaps_gt_ = tf.reshape(target, [-1, 17, 64*48])

        heatmaps_pred = tf.split(heatmaps_, 17, 1)
        heatmaps_gt = tf.split(heatmaps_gt_, 17, 1)

        batch_size = tf.shape(imgs)[0]
        loss_list = []
        idx = 0

        for i in range(17):
            heatmap_pred = tf.squeeze(heatmaps_pred[i])#batchsize * 3072
            heatmap_gt = tf.squeeze(heatmaps_gt[i])#batchsize * 3072
            if cfg.use_target_weight:
                
                loss = tf.nn.l2_loss((tf.multiply(heatmap_pred, target_weight[:,idx]) - tf.multiply(heatmap_gt, target_weight[:,idx]))) / tf.cast(batch_size, tf.float32)
                # loss = tf.square(tf.multiply(heatmaps, target_weight[:,idx]) - tf.multiply(heatmapt_, target_weight[:,idx]))*0.5
            else:
                # pdb.set_trace()
                loss = tf.nn.l2_loss(heatmap_pred - heatmap_gt) / tf.cast(batch_size, tf.float32)
                # loss = tf.square(heatmaps - heatmapt_)*0.5
            loss_list.append(loss)
            idx+=1
        # pdb.set_trace()
        total_loss = tf.add_n(loss_list, name='total_loss')

        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
        else:
            wd_cost = tf.constant(0.0)
    
        self.cost = tf.add_n([total_loss, wd_cost], name='cost')


        # ========================== Summary & Outputs ==========================
        tf.summary.image(name='image', tensor=imgs, max_outputs=3)
        # tf.summary.image(name='mask', tensor=target, max_outputs=3)

        # stage1_output1 = tf.identity(heatmap_outputs[0],  name = 'heatmaps_1')
        # stage1_output2 = tf.identity(paf_outputs[0], name = 'pafs_1')
        # stage2_output1 = tf.identity(heatmap_outputs[1],  name = 'heatmaps_2')
        # stage2_output2 = tf.identity(paf_outputs[1], name = 'pafs_2')
        # stage3_output1 = tf.identity(heatmap_outputs[2],  name = 'heatmaps_3')
        # stage3_output2 = tf.identity(paf_outputs[2], name = 'pafs_3')
        # stage4_output1 = tf.identity(heatmap_outputs[3],  name = 'heatmaps_4')
        # stage4_output2 = tf.identity(paf_outputs[3], name = 'pafs_4')
        # stage5_output1 = tf.identity(heatmap_outputs[4],  name = 'heatmaps_5')
        # stage5_output2 = tf.identity(paf_outputs[4], name = 'pafs_5')

       

        add_moving_summary(self.cost)
        add_moving_summary(total_loss)
        
        for idx, sub_loss in enumerate(loss_list):
            add_moving_summary(tf.identity(loss_list[idx], name='stage%d_l2_loss' % (idx+1)))

        targetss=  tf.transpose(target, [0,2,3,1])
        logitss = tf.transpose(logits, [0, 2, 3, 1])

        
        gt_heatmap_shown = tf.reduce_max(targetss, axis=3, keep_dims=True)
        tf.summary.image(name='gt_heatmap', tensor=gt_heatmap_shown, max_outputs=3)


       
        heatmap_shown = tf.reduce_max(logitss, axis=3, keep_dims=True)
        tf.summary.image(name='heatmap', tensor=heatmap_shown, max_outputs=3)
        
        

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', cfg.base_lr, summary=True)
        # return tf.train.MomentumOptimizer(learning_rate=lr, momentum=cfg.momentum, use_nesterov=True)
        return tf.train.AdamOptimizer(lr)

def get_data(train_or_test, batch_size):
    is_train = train_or_test == 'train'

    ds = Data(train_or_test)
    sample_num = ds.size()

    if is_train:
        augmentors = [
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045][::-1],
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            # imgaug.Clip(),
            imgaug.ToUint8()
        ]

    else:
        augmentors = [
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors)

    if is_train:
        ds = PrefetchDataZMQ(ds, min(8, multiprocessing.cpu_count()))
    ds = BatchData(ds, batch_size, remainder = not is_train)
    return ds, sample_num

def get_config(args, model):
    ds_train, sample_num = get_data('train', args.batch_size_per_gpu)
    ds_val, _ = get_data('val', args.batch_size_per_gpu)

    return TrainConfig(
        dataflow = ds_train,
        callbacks = [
            ModelSaver(),
            # PeriodicTrigger(InferenceRunner(ds_val, [ScalarStats('cost')]),
            #                 every_k_epochs=2),
            HumanHyperParamSetter('learning_rate'),
        ],
        model = model,
        max_epoch = 200,
        steps_per_epoch = sample_num // (args.batch_size_per_gpu * get_nr_gpu()),
        # steps_per_epoch = 20,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--batch_size_per_gpu', type=int, default=16)
    parser.add_argument('--logdir', help="directory of logging", default=None)
   



    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model()
        
    if args.logdir != None:
        logger.set_logger_dir(os.path.join("train_log", args.logdir))
    else:
        logger.auto_set_dir()

    config = get_config(args, model)
    if args.load:
        config.session_init = get_model_loader(args.load)
    
    trainer = SyncMultiGPUTrainerParameterServer(get_nr_gpu())
    launch_train_with_config(config, trainer)
