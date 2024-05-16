import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import linen as nn
from flax.training import train_state
from time import time
from typing import Callable, Sequence

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

class MLPModule(nn.Module):
  """A FLAX module for a simple multi layer perceptron."""
  n_features: Sequence[int]
  activation: Callable
  @nn.compact
  def __call__(self, x):
    for i, n_features in enumerate(self.n_features):
      x = nn.Dense(n_features)(self.activation(x) if i > 0 else x)
    return x

@jax.jit
def apply_model(state, X, y):
  """Compute gradients, loss and accuracy."""
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, X)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
    return loss, logits
  (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == y)
  return grads, loss, accuracy

@jax.jit
def update_model(state, grads):
  """Update parameters with pre-computed gradients."""
  return state.apply_gradients(grads=grads)

@jax.jit
def predict_logits(state, X):
  return state.apply_fn({'params': state.params}, X)

class MLPClassifier(BaseEstimator, ClassifierMixin):
  """A simple multi layer perceptron.
  
  Args:
      TODO
  """
  def __init__(
      self,
      n_features = (64,),
      n_epochs = 2000,
      val_size = .1,
      batch_size = 256,
      lr_init = 1e-2,
      lr_steps = {250: .5, 500: .5, 750: .5, 1000: .5, 1250: .5, 1500: .5, 1750: .5},
      momentum = .9,
      activation = "tanh",
      random_state = None,
      n_epochs_between_val = 10,
      verbose = False,
      ):
    self.n_features = n_features
    self.n_epochs = n_epochs
    self.val_size = val_size
    self.batch_size = batch_size
    self.lr_init = lr_init
    self.lr_steps = lr_steps
    self.momentum = momentum
    self.activation = activation
    self.random_state = random_state
    self.n_epochs_between_val = n_epochs_between_val
    self.verbose = verbose
  def fit(self, X, y):
    self.random_state = np.random.RandomState(self.random_state)
    X_trn, X_val, y_trn, y_val = train_test_split( # split a validation set
      X,
      y,
      test_size = self.val_size,
      stratify = y,
      random_state = self.random_state,
    )

    # instantiate the model
    if self.activation == "tanh":
      activation = nn.activation.tanh
    elif self.activation == "sigmoid":
      activation = nn.activation.sigmoid
    elif self.activation == "relu":
      activation = nn.activation.relu
    else:
      raise ValueError(f"Unknown activation={self.activation}")
    module = MLPModule(np.concatenate((self.n_features, [len(np.unique(y))])), activation)
    self.state = train_state.TrainState.create(
      apply_fn = module.apply,
      params = module.init( # initialize parameters
        jax.random.key(self.random_state.randint(np.iinfo(np.uint32).max)),
        X[[1]] # a dummy batch with one sample
      )["params"],
      tx = optax.sgd(
        learning_rate = optax.piecewise_constant_schedule(
          init_value = self.lr_init,
          boundaries_and_scales = self.lr_steps,
        ),
        momentum = self.momentum
      )
    )

    # take out the training
    progress = {
      "epoch": [],
      "loss_trn": [],
      "loss_val": [],
      "acc_val": [],
      "time": [],
    }
    t_0 = time()
    for epoch_index in range(self.n_epochs): # take out one epoch
      batch_losses = []
      i_epoch = np.random.default_rng(epoch_index).permutation(len(y_trn)) # shuffle

      # mini-batch training
      for batch_index in range(len(y_trn) // self.batch_size):
        i_batch = i_epoch[batch_index * self.batch_size:(batch_index+1) * self.batch_size]
        grads, loss, _ = apply_model(self.state, X_trn[i_batch], y_trn[i_batch])
        self.state = update_model(self.state, grads) # update the training state
        batch_losses.append(loss)

      # validation
      if (epoch_index+1) % self.n_epochs_between_val == 0:
        progress["epoch"].append(epoch_index+1)
        progress["loss_trn"].append(np.mean(batch_losses))
        _, loss_val, acc_val = apply_model(self.state, X_val, y_val) # validate
        progress["loss_val"].append(loss_val)
        progress["acc_val"].append(acc_val)
        progress["time"].append(time() - t_0)
        if self.verbose:
          print(
            f"[{epoch_index+1:3d}/{self.n_epochs}]",
            f"loss_trn={progress['loss_trn'][-1]:.5f}",
            f"loss_val={progress['loss_val'][-1]:.5f}",
            f"acc_val={progress['acc_val'][-1]:.5f}",
            f"t={progress['time'][-1]:.1f}s"
          )
    self.progress_ = pd.DataFrame(progress)
    return self
  def predict_proba(self, X):
    return nn.activation.softmax(predict_logits(self.state, X), axis=1)
  def predict(self, X):
    return predict_logits(self.state, X).argmax(axis=1)
