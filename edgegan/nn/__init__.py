from .modules import *
import tensorflow as tf

image_summary = tf.compat.v1.summary.image
scalar_summary = tf.compat.v1.summary.scalar
histogram_summary = tf.compat.v1.summary.histogram
merge_summary = tf.compat.v1.summary.merge
SummaryWriter = tf.compat.v1.summary.FileWriter
