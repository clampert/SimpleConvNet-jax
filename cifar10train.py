#!/usr/bin/env python3
# coding: utf-8

### simple CIFAR10 training (ideally fast) in jax

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
import numpy as np

from functools import partial
from tqdm import tqdm
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from flax.training import train_state
from flax import linen as nn

from cifar10model import SpeedyResNet
from cifar10data import load_data, prepare_ds

config = {
  'seed': int(os.getenv("SEED", 0)),
  'batch_size': int(os.getenv("BATCH_SIZE", 512)),
  'val_batch_size': int(os.getenv("BATCH_SIZE", 1000)),
  'test_batch_size': int(os.getenv("BATCH_SIZE", 1000)),
  'num_epochs': int(os.getenv("NUM_EPOCHS", 24)),
  'warmup': float(os.getenv("WARMUP", 0.1)),
  'learning_rate': float(os.getenv("LEARNING_RATE", 0.05)),
  'momentum': float(os.getenv("MOMENTUM", 0.9)),
  'nesterov': bool(os.getenv("NESTEROV", True)),
  'weight_decay': float(os.getenv("WEIGHT_DECAY", 0.005)),
  'ntrain': int(os.getenv("NTRAIN", 45000)),
  'nval': int(os.getenv("NVAL", 5000)),
  'dtype': jnp.float16,
  'base_depth': 64, ## should be a factor of 8
  'temperature': 8.
}

# subclass to handle additional batchnorm data
class TrainState(train_state.TrainState):
  batch_stats: flax.core.FrozenDict[str, Any]
  pass

def create_train_state(model, learning_rate_fn):
  """Creates an initial `TrainState`."""
  variables = model.init(jax.random.PRNGKey(config['seed']), jnp.ones([1, 32, 32, 3]))
  params = variables['params']
  batch_stats = variables['batch_stats']
  
  tx = optax.chain(
    optax.add_decayed_weights(config['weight_decay']), # explicit weight decay
    optax.sgd(learning_rate_fn, momentum=config['momentum'], nesterov=config['nesterov'])
  )
  return TrainState.create(apply_fn=model.apply, params=params, batch_stats=batch_stats, tx=tx)

@partial(jax.jit, static_argnames=['learning_rate_fn'])
def train_step(state, batch, learning_rate_fn):
  """Train for a single step."""
  image, labels = batch[0], batch[1]
  
  def loss_fn(params): # with temperature scaling
    logits, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, x=image, train=True, mutable=['batch_stats'])
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits/config['temperature'], labels=labels))
    loss = config['temperature']*loss
    return loss, (logits, updates)
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (logits, updates)), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  state = state.replace(batch_stats=updates['batch_stats'])
  
  acc = jnp.mean(jnp.argmax(logits, -1) == labels)
  lr = learning_rate_fn(state.step)
  metrics = {'loss': loss, 'accuracy': acc, 'lr':lr}
  return state, metrics

@jax.jit
def compute_metrics(state, batch):
  image,labels = batch[0],batch[1]
  logits = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, image, train=False)
  loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels), dtype=np.float32) # avoids troube with numpy
  metrics = {'loss': loss, 'accuracy': jnp.mean(jnp.argmax(logits, -1) == labels, dtype=jnp.float32) }
  return metrics

def accuracy(state, ds):
  total_loss,total_accuracy = [],[]
  for batch in ds.as_numpy_iterator():
    metrics = compute_metrics(state=state, batch=batch)
    total_loss.append(metrics['loss'])
    total_accuracy.append(metrics['accuracy'])
  return np.mean(total_loss),np.mean(total_accuracy) # avg of batch-averages. Not exact but good enough here


def main():
  train_ds, val_ds, _ = load_data(ntrain=config['ntrain'], nval=config['nval'], ntest=0)
  if val_ds is not None:
    val_ds = prepare_ds(val_ds, batch_size=config['test_batch_size'])
  num_steps_per_epoch = int(train_ds.cardinality())//config['batch_size']
  num_epochs = config['num_epochs']

  print(f"Running {num_steps_per_epoch} steps per epoch with batchsize {config['batch_size']} for {num_epochs} epochs.")

  model = SpeedyResNet(dtype=config['dtype'])
  dummy_input = jnp.ones((1, 32, 32, 3), dtype=config['dtype'])
  print(model.tabulate(jax.random.PRNGKey(0), dummy_input, console_kwargs={"width": 144}))

  learning_rate_fn = optax.warmup_cosine_decay_schedule(
    init_value=0,
    peak_value=config['learning_rate'],
    warmup_steps=config['warmup']*num_epochs*num_steps_per_epoch,
    decay_steps=num_epochs*num_steps_per_epoch
  )
  state = create_train_state(model=model, learning_rate_fn=learning_rate_fn)

  today = datetime.datetime.now().isoformat().replace(':','-')
  CKPT_DIR = f'{os.getcwd()}/snapshot/cifar10-{today}/'
  print(f"Writing snapshots to directory {CKPT_DIR}")

  checkpoint_options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
  with ocp.CheckpointManager(
    ocp.test_utils.erase_and_create_empty(CKPT_DIR),
    options=checkpoint_options,
  ) as checkpoint_manager:

    for epoch in tqdm(range(num_epochs)):
      ds = prepare_ds(
            train_ds, 
            batch_size = config['batch_size'], 
            do_shuffle = True, 
            do_augment = True
      ) # new shuffle and augmentation every epoch
      
      for step,batch in enumerate(ds.as_numpy_iterator()):
        state, metrics = train_step(state, batch, learning_rate_fn)
      
      acc = accuracy(state, val_ds) if val_ds is not None else (0,0)
      print(f"train epoch: {epoch} trn ({metrics['loss']:.6f}, {metrics['accuracy']:.6f})  val {acc} lr {metrics['lr']:.6f}")
      
      ckpt = {'state': state, 'config': None, 'data': None}
      checkpoint_manager.save(epoch, args=ocp.args.StandardSave(ckpt))
    
    print("Finished training")
  
if __name__ == "__main__":
  main()
