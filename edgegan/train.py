import json
import os

import numpy as np
import tensorflow as tf
from numpy.random import seed

from edgegan.models import EdgeGAN
from edgegan.utils import makedirs, pp
from edgegan.utils.data import Dataset


_FLAGS = tf.compat.v1.flags
_FLAGS.DEFINE_string("gpu", "0", "Gpu ID")
_FLAGS.DEFINE_string("name", "edgegan", "Folder for all outputs")
_FLAGS.DEFINE_string("outputsroot", "outputs", "Outputs root")
_FLAGS.DEFINE_integer("epoch", 100, "Epoch to train [25]")
_FLAGS.DEFINE_float("learning_rate", 0.0002, "")
_FLAGS.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
_FLAGS.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
_FLAGS.DEFINE_integer(
    "input_height", 64, "The size of image to use (will be center cropped). [108]")
_FLAGS.DEFINE_integer(
    "input_width", 128, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
_FLAGS.DEFINE_integer("output_height", 64,
                      "The size of the output images to produce [64]")
_FLAGS.DEFINE_integer("output_width", 128,
                      "The size of the output images to produce. If None, same value as output_height [None]")
_FLAGS.DEFINE_string("dataset", "class14", "")
_FLAGS.DEFINE_string("input_fname_pattern", "*png",
                     "Glob pattern of filename of input images [*]")
_FLAGS.DEFINE_string("checkpoint_dir", None, "")
_FLAGS.DEFINE_string("logdir", None, "")
_FLAGS.DEFINE_string("dataroot", "./data", "Root directory of dataset [data]")
_FLAGS.DEFINE_integer("save_checkpoint_frequency", 500,
                      "frequency for saving checkpoint")
_FLAGS.DEFINE_boolean("crop", False, "")


# weight of loss
_FLAGS.DEFINE_float("stage1_zl_loss", 10.0, "weight of z l1 loss")

# multi class
_FLAGS.DEFINE_boolean("multiclasses", True, "if use focal loss")
_FLAGS.DEFINE_integer("num_classes", 14, "num of classes")
_FLAGS.DEFINE_string("SPECTRAL_NORM_UPDATE_OPS",
                     "spectral_norm_update_ops", "")

_FLAGS.DEFINE_boolean("if_resnet_e", True, "if use resnet for E")
_FLAGS.DEFINE_boolean("if_resnet_g", False, "if use resnet for G")
_FLAGS.DEFINE_boolean("if_resnet_d", False, "if use resnet for D")
_FLAGS.DEFINE_float("lambda_gp", 10.0, "")

_FLAGS.DEFINE_string("E_norm", "instance",
                     "normalization options:[instance, batch, norm]")
_FLAGS.DEFINE_string("G_norm", "instance",
                     "normalization options:[instance, batch, norm]")
_FLAGS.DEFINE_string("D_norm", "instance",
                     "normalization options:[instance, batch, norm]")


_FLAGS.DEFINE_boolean("use_image_discriminator", True,
                      "True for using patch discriminator, modify the size of input of discriminator")
_FLAGS.DEFINE_integer("image_dis_size", 128, "The size of input for image discriminator")
_FLAGS.DEFINE_boolean("use_edge_discriminator", True,
                      "True for using patch discriminator, modify the size of input of discriminator, user for edge discriminator when G_num == 2")
_FLAGS.DEFINE_integer("edge_dis_size", 128, "The size of input for edge discriminator")
_FLAGS.DEFINE_float("joint_dweight", 1.0,
                    "weight of origin discriminative loss")
_FLAGS.DEFINE_float("image_dweight", 1.0,
                    "weight of image discriminative loss, is ineffective when use_image_discriminator is false")
_FLAGS.DEFINE_float("edge_dweight", 1.0,
                    "weight of edge discriminative loss, is ineffective when use_edge_discriminator is false")
_FLAGS.DEFINE_integer("z_dim", 100, "dimension of random vector z")
FLAGS = _FLAGS.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

def make_outputs_dir(flags):
    makedirs(flags.outputsroot)
    makedirs(flags.checkpoint_dir)
    makedirs(flags.logdir)


def update_flags(flags):
    if flags.input_width is None:
        flags.input_width = flags.input_height
    if flags.output_width is None:
        flags.output_width = flags.output_height

    if not flags.multiclasses:
        flags.num_classes = None

    path = os.path.join(flags.outputsroot, flags.name)
    setattr(flags, 'checkpoint_dir', os.path.join(path, 'checkpoints'))
    setattr(flags, 'logdir', os.path.join(path, 'logs'))

    return flags

def save_flags(flags):
    path = os.path.join(flags.outputsroot, flags.name)

    flag_dict = flags.flag_values_dict()
    with open(os.path.join(path, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f, indent=4)

    return flags

def main(_):
    phase = 'train'
    flags = update_flags(FLAGS)
    pp.pprint(flags.__flags)
    make_outputs_dir(flags)
    save_flags(flags)

    run_config = tf.compat.v1.ConfigProto()
    run_config.gpu_options.allow_growth = True
    dataset_config = {
        'input_height': flags.input_height,
        'input_width': flags.input_width,
        'output_height': flags.output_height,
        'output_width': flags.output_width,
        'crop': flags.crop,
        'grayscale': False,
        'z_dim': flags.z_dim,
    }

    with tf.compat.v1.Session(config=run_config) as sess:
        dataset = Dataset(
            flags.dataroot, flags.dataset,
            flags.train_size, flags.batch_size,
            dataset_config, flags.num_classes, phase)
        edgegan_model = EdgeGAN(sess, flags, dataset, z_dim=flags.z_dim)
        edgegan_model.train()


if __name__ == '__main__':
    tf.compat.v1.app.run()
