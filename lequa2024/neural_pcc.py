import flax
import jax
import jaxopt
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

class LRModule(nn.Module):
  """A FLAX module for a simple multi layer perceptron."""
  n_classes: int
  @nn.compact
  def __call__(self, x):
    return nn.Dense(self.n_classes)(x)

class NeuralPCC(BaseEstimator, ClassifierMixin):
  """A logistic regression with set-based training."""
  def __init__(
      self,
      set_frac = .5,
      C = 1e0,
      tol = 1e-4,
      # n_features = (256,),
      max_iter = 100,
      # val_size = .1,
      # activation = "tanh",
      random_state = None,
      set_training = None,
      verbose = False,
      ):
    self.set_frac = set_frac
    self.C = C
    self.tol = tol
    # self.n_features = n_features
    self.max_iter = max_iter
    # self.val_size = val_size
    # self.activation = activation
    self.random_state = random_state
    self.set_training = set_training
    self.verbose = verbose
  def fit(self, X, y):
    self.random_state = np.random.RandomState(self.random_state)
    self.module = LRModule(len(np.unique(y))) # logistic regression

    def loss_fn(params):
      logits = self.module.apply({"params": params}, X)
      loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y).mean()
      if self.set_training is not None:
        p_est = jnp.mean(
          nn.activation.softmax(
            self.module.apply({'params': params}, self.set_training.X),
            axis = 2
          ),
          axis = 1
        )
        mse = jnp.sum((p_est - self.set_training.p)**2, axis=1).mean()
        loss = (1-self.set_frac) * loss + self.set_frac * mse
      if self.C:
        loss += jaxopt.tree_util.tree_l2_norm(params, squared=True) / 2 / self.C
      return loss

    # prepare the optimizer
    solver = jaxopt.LBFGS(loss_fn, maxiter=self.max_iter, tol=self.tol)
    self.params = self.module.init( # initialize parameters
      jax.random.key(self.random_state.randint(np.iinfo(np.uint32).max)),
      X[[1]] # a dummy batch with one sample
    )["params"]
    t_0 = time()
    self.params, state = solver.run(self.params)
    if self.verbose:
      print(f"L={state.value:.5f} t={time() - t_0:.1f}s")
    return self
  def predict_proba(self, X):
    return nn.activation.softmax(self._predict_logits(X), axis=1)
  def predict(self, X):
    return self._predict_logits(X).argmax(axis=1)
  def _predict_logits(self, X):
    return self.module.apply({"params": self.params}, X)

class SetTraining():
  def __init__(self, X, p):
    self.X = X
    self.p = p
