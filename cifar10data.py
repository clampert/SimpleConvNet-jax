#!/usr/bin/env python3
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from functools import partial

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU") # forbid TF to allocate GPU memory
import tensorflow_datasets as tfds


def _window_mask(bbox, image_size):
  """Mask a part of the image."""
  y, x, h, w = bbox
  y_range = tf.reshape(tf.range(image_size[0]), [image_size[0], 1, 1])
  x_range = tf.reshape(tf.range(image_size[1]), [1, image_size[1], 1])
  mask = tf.logical_and(tf.logical_and(y <= y_range, y_range < y + h),
                        tf.logical_and(x <= x_range, x_range < x + w))
  return tf.cast(mask, tf.float32)

def _random_box(image_size, box_size):
  """Sample a box of size box_size uniformly."""
  y = tf.random.uniform(shape=(), minval=0, maxval=image_size[0]-box_size[0], dtype=tf.int32)
  x = tf.random.uniform(shape=(), minval=0, maxval=image_size[1]-box_size[1], dtype=tf.int32)
  return (x, y)+box_size


def _cutout(image, box_size):
  """Zero out part of the image."""
  image_size = image.shape
  bbox = _random_box(image_size, box_size)
  mask = _window_mask(bbox, image.shape)
  return image*(1-mask)


def normalize_img(image, cifar10_mean=0.5, cifar10_std=1.):
    image = tf.cast(image, tf.float32) / 255.
    image = (image - cifar10_mean)/cifar10_std
    return image


def augment_image(image, label, box_size=(8,8)):
  image = tf.pad(image, paddings=[[4,4],[4,4],[0,0]], mode="REFLECT")
  image = tf.image.random_crop(value=image, size=(32, 32, 3))
  image = tf.image.random_flip_left_right(image=image)
  image = _cutout(image, box_size)
  return image, label

def process_sample(image, label):
  image = normalize_img(image) # also converts to float
  return image, label

def prepare_ds(ds, batch_size=1000, do_shuffle=False, do_augment=False):
  ds = ds.cache()
  if do_shuffle:
    ds = ds.shuffle(buffer_size=ds.cardinality())
  ds = ds.map(process_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  if do_augment:
    ds = ds.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(10)
  return ds

def load_data(ntrain=50000, nval=0, ntest=10000):
  train_ds = tfds.load('cifar10', split=f'train[:{ntrain}]', as_supervised=True) if ntrain else None
  val_ds = tfds.load('cifar10', split=f'train[{ntrain}:{ntrain+nval}]', as_supervised=True) if nval else None
  test_ds = tfds.load('cifar10', split=f'test[:{ntest}]', as_supervised=True) if ntest else None
  return train_ds, val_ds, test_ds
