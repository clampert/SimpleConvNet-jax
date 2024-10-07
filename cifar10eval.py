#!/usr/bin/env python3
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from cifar10model import SpeedyResNet
from cifar10data import load_data, prepare_ds

def accuracy(model, ds):
    acc = []
    for batch in ds.as_numpy_iterator():
        image, labels = batch[0], batch[1]
        logits = model(image)
        acc.extend(jnp.argmax(logits, -1) == labels)
    return np.mean(acc)

def main():
  try:
      CKPT_DIR = sys.argv[1]
  except IndexError:
      print(f"Usage: {sys.argv[0]} checkpointpath")
      raise SystemExit

  with ocp.CheckpointManager(CKPT_DIR) as checkpoint_manager:
      ckpt = checkpoint_manager.restore(
          checkpoint_manager.latest_step(),
          args=ocp.args.StandardRestore(None),
      )
      state = ckpt['state']

  model = SpeedyResNet()
  model.apply = jax.jit(partial(model.apply, train=False)) # precompile
  model = model.bind({'params': state['params'], 'batch_stats': state['batch_stats']})

  train_ds, val_ds, test_ds = load_data(ntrain=45000, nval=5000, ntest=10000)
  train_ds, val_ds, test_ds = prepare_ds(train_ds),prepare_ds(val_ds),prepare_ds(test_ds)

  acc = accuracy(model, train_ds)
  print(f"accuracy[%] train: {100*acc:.3f}")
  acc = accuracy(model, val_ds)
  print(f"accuracy[%] val: {100*acc:.3f}")
  acc = accuracy(model, test_ds)
  print(f"accuracy[%] test: {100*acc:.3f}")

if __name__ == "__main__":
  main()
