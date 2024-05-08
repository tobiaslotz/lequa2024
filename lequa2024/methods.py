import jax
import jax.numpy as jnp
import numpy as np
import traceback
from abc import abstractmethod
from scipy.optimize import minimize
from scipy.stats import scoreatpercentile
from sklearn.base import clone
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_is_fitted, NotFittedError
from sklearn.model_selection import cross_val_predict
from quapy.data.base import LabelledCollection
from quapy.method.base import BaseQuantifier
from qunfold.methods import (
  _CallbackState,
  _check_derivative,
  _rand_x0,
  DerivativeError,
  Result,
)
from qunfold.transformers import _check_y, class_prevalences, ClassTransformer

def _bw_scott(X):
  sigma = np.std(X, ddof=1)
  return 3.49 * sigma * X.shape[0]**(-0.333)

def _bw_silverman(X):
  norm_iqr = (scoreatpercentile(X, 75) - scoreatpercentile(X, 25)) / 1.349
  sigma = np.std(X, ddof=1)
  A = np.minimum(sigma, norm_iqr) if norm_iqr > 0 else sigma
  return 0.9 * A * X.shape[0]**(-0.2)

class KDEBase(BaseQuantifier):
  """Abstract base class of the KDEy method by González-Moreo et al. (2024)."""
  def __init__(
      self,
      classifier,
      bandwidth = 0.1,
      fit_classifier = True,
      random_state = None,
      solver = 'SLSQP',
      n_cross_val = 10
      ) -> None:
    self.classifier = classifier
    self.bandwidth = bandwidth
    self.random_state = random_state
    self.solver = solver
    self.fit_classifier = fit_classifier
    self.n_cross_val = n_cross_val
  def fit(self, X, y=None, n_classes=None):
    if y is None:
      return self.fit(*X.Xy, X.n_classes) # assume that X is a QuaPy LabelledCollection
    _check_y(y, n_classes)
    self.p_trn = class_prevalences(y, n_classes)
    self.n_classes = len(self.p_trn) # not None anymore
    #self.preprocessor = ClassTransformer(classifier=self.classifier, is_probabilistic=True, fit_classifier=fit_classifier)
    #fX, _ = self.preprocessor.fit_transform(X, y, average=False)
    fX = cross_val_predict(self.classifier, X, y, cv=self.n_cross_val, method='predict_proba')
    if self.fit_classifier or not self.__classifier_fitted():
      self.classifier.fit(X, y)
    if isinstance(self.bandwidth, list) or isinstance(self.bandwidth, np.ndarray):
      assert len(self.bandwidth) == self.n_classes, (
        f"bandwidth must either be a single scalar or a sequence of length n_classes.\n"
        f"Received {len(self.bandwidth)} values for bandwidth, but dataset has {n_classes} classes."
      )
      self.mixture_components = [
          KernelDensity(bandwidth=self.bandwidth[c]).fit(fX[y==c])
          for c in range(self.n_classes)
        ]
    else:
      self.mixture_components = [
          KernelDensity(bandwidth=self.bandwidth).fit(fX[y==c])
          for c in range(self.n_classes)
        ]
    return self
  def predict(self, X):
    fX = self.classifier.predict_proba(X)
    return self.solve(fX)
  @abstractmethod
  def solve(self, fX):
    pass
  def quantify(self, X):
      return self.predict(X)
  def __classifier_fitted(self):
    try:
      check_is_fitted(self.classifier)
      return True
    except NotFittedError:
      return False


class KDEyMLQP(KDEBase):
  """The Maximum-Likelihood variant of the KDEy method by González-Moreo et al. (2024)."""
  def __init__(
      self,
      classifier,
      bandwidth = 0.1,
      fit_classifier = True,
      random_state = None,
      solver = 'SLSQP',
      n_cross_val = 10
      ) -> None:
    KDEBase.__init__(
      self,
      classifier = classifier,
      bandwidth = bandwidth,
      random_state = random_state,
      solver = solver,
      n_cross_val=n_cross_val,
      fit_classifier=fit_classifier,
    )
  def solve(self, fX):
    np.random.RandomState(self.random_state)
    epsilon = 1e-10
    n_classes = len(self.mixture_components)
    test_densities = [np.exp(mc.score_samples(fX)) for mc in self.mixture_components]
    def neg_loglikelihood(prevs):
      test_mixture_likelihood = sum(prev_i * dens_i for prev_i, dens_i in zip(prevs, test_densities))
      test_loglikelihood = np.log(test_mixture_likelihood + epsilon)
      return -np.sum(test_loglikelihood)
    x0 = np.full(fill_value=1 / n_classes, shape=(n_classes,))
    bounds = tuple((0, 1) for _ in range(n_classes))
    constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
    opt = minimize(
      neg_loglikelihood,
      x0=x0,
      method=self.solver,
      bounds=bounds,
      constraints=constraints
    )
    return Result(opt.x, opt.nit, opt.message)

######## code of git@github.com:mirkobunse/acspy.git ########

def n_samples_per_class(y, n_classes=None):
  """Determine the number of instances per class.

  Args:
      y: An array of labels, shape (n_samples,).
      n_classes (optional): The number of classes. Defaults to `None`, which corresponds to `len(np.unique(y))`.

  Returns:
      An array of label counts, shape (n_classes,).
  """
  if n_classes is None:
    n_classes = len(np.unique(y))
  n_samples_per_class = np.zeros(n_classes, dtype=int)
  i, n = np.unique(y, return_counts=True)
  n_samples_per_class[i] = n # non-existing classes maintain a zero entry
  return n_samples_per_class

def draw_indices(
    y,
    score,
    m = None,
    n_classes = None,
    allow_duplicates = True,
    allow_others = False,
    allow_incomplete = False,
    min_samples_per_class = 0,
    random_state = None,
  ):
  """Draw m indices of instances, according to the given class scores.

  Args:
      y: An pool of labels, shape (n_samples,).
      score: An array of utility scores for the acquisition of each clas, shape (n_classes,).
      m (optional): The total number of instances to be drawn. Defaults to `None`, which is only valid for binary classification tasks and which corresponds to drawing the maximum number of instances for which the scores can be fulfilled.
      n_classes (optional): The number of classes. Defaults to `None`, which corresponds to `len(np.unique(y))`.
      allow_duplicates (optional): Whether to allow drawing one sample multiple times. Defaults to `False`.
      allow_others (optional): Whether to allow drawing other classes if the desired class is exhausted. Defaults to `True`.
      allow_incomplete (optional): Whether to allow an incomplete draw if all classes are exhausted. Defaults to `True`.
      min_samples_per_class (optional): The minimum number of samples per class. Defaults to `0`.
      random_state (optional): A numpy random number generator, or a seed thereof. Defaults to `None`, which corresponds to `np.random.default_rng()`.

  Returns:
      An array of indices, shape (m,).
  """
  if n_classes is None:
    n_classes = len(np.unique(y))
  elif len(score) != n_classes:
    raise ValueError("len(score) != n_classes")
  random_state = np.random.default_rng(random_state)
  m_pool = n_samples_per_class(y, n_classes) # number of available samples per class
  p = np.maximum(0, score) / np.maximum(0, score).sum() # normalize scores to probabilities
  if not np.isfinite(p).all():
    raise ValueError(f"NaN probabilities caused by score={score}")

  # determine the number of instances to acquire from each class
  if m is not None:
    to_take = np.maximum(
      np.round((m - min_samples_per_class*len(p))*p).astype(int),
      min_samples_per_class
    )
    if not allow_duplicates:
      to_take = np.minimum(m_pool, to_take)
    while m != to_take.sum(): # rarely takes more than one iteration
      m_remaining = m - to_take.sum()
      if m_remaining > 0: # are additional draws needed?
        i = np.nonzero(to_take < m_pool)[0]
        if len(i) == 0:
          if allow_incomplete:
            break
          else:
            raise ValueError(
              f"All classes are exhausted; consider setting allow_incomplete=True"
            )
        elif allow_others or allow_duplicates or len(i) == len(p):
          bincount = np.bincount(random_state.choice(
            len(i),
            size = m_remaining,
            p = np.maximum(1/m, p[i]) / np.maximum(1/m, p[i]).sum(),
          ))
          to_take[i[:len(bincount)]] += bincount
        else:
          raise ValueError(
            f"Class {np.setdiff1d(np.arange(len(p)), i)[0]} exhausted; "
            "consider setting allow_others=True "
          )
      elif m_remaining < 0: # are less draws needed?
        i = np.nonzero(to_take > min_samples_per_class)[0]
        bincount = np.bincount(random_state.choice(
          len(i),
          size = -m_remaining, # negate to get a positive value
          p = np.maximum(1/m, 1-p[i]) / np.maximum(1/m, 1-p[i]).sum(),
        ))
        to_take[i[:len(bincount)]] -= bincount
      to_take = np.maximum(min_samples_per_class, np.minimum(to_take, m_pool))
  else: # m is None
    if len(m_pool) != 2:
      raise ValueError("m=None is only valid for binary classification tasks")
    to_take = m_pool.copy()
    if p[0] > m_pool[0] / m_pool.sum(): # do we have to increase the class "0" probability?
      to_take[1] = int(np.round(m_pool[0] * p[1] / p[0])) # sub-sample class "1"
    else: # otherwise, we need to increase the class "1" probability
      to_take[0] = int(np.round(m_pool[1] * p[0] / p[1])) # sub-sample class "0"
    to_take = np.minimum(to_take, m_pool)

  # draw indices that represent instances of the classes to acquire
  i_rand = random_state.permutation(len(y)) # random order after shuffling
  i_draw = [] # array of drawn index arrays (relative to i_rand)
  for c in range(n_classes):
    i_c = np.arange(len(y))[y[i_rand] == c]
    if to_take[c] <= len(i_c):
      i_draw.append(i_c[:to_take[c]])
    elif allow_duplicates:
      i_c = np.tile(i_c, int(np.ceil(to_take[c] / len(i_c)))) # repeat i_c multiple times
      i_draw.append(i_c[:to_take[c]]) # draw from the tiled array
    else:
      raise ValueError(
        f"Class {c} exhausted; consider setting allow_duplicates=True "
        f"({to_take[c]} requested, {len(i_c)} available)"
      )
  return i_rand[np.concatenate(i_draw)]

######## end of git@github.com:mirkobunse/acspy.git  ########

# generalize our softmax "trick" from l[0]=0 to l[i]=0 with any i
def _jnp_softmax(l, i):
  exp_l = jnp.ones(len(l)+1)
  exp_l = exp_l.at[jnp.setdiff1d(jnp.arange(len(exp_l)), jnp.array([i]))].set(jnp.exp(l))
  return exp_l / exp_l.sum()
def _np_softmax(l, i):
  exp_l = np.ones(len(l)+1)
  exp_l[np.setdiff1d(np.arange(len(exp_l)), np.array([i]))] = np.exp(l)
  return exp_l / exp_l.sum()

class EMaxL(BaseQuantifier):
  """Our maximum likelihood fusion ensemble."""
  def __init__(
      self,
      base_estimator,
      n_estimators = 1,
      random_state = None,
      min_samples_per_class = 5,
      solver = "trust-ncg",
      solver_options = {"gtol": 0, "maxiter": 200}, # , "disp": True
      tau = 0,
      multistart = False,
      ) -> None:
    self.base_estimator = base_estimator
    self.n_estimators = n_estimators
    self.random_state = random_state
    self.min_samples_per_class = min_samples_per_class
    self.solver = solver
    self.solver_options = solver_options
    self.tau = tau
    self.multistart = multistart
  def fit(self, X, y=None, n_classes=None):
    if y is None:
      return self.fit(*X.Xy, X.n_classes) # assume that X is a QuaPy LabelledCollection
    _check_y(y, n_classes)
    self.p_trn = class_prevalences(y, n_classes)
    self.n_classes = len(self.p_trn) # not None anymore
    rng = np.random.RandomState(self.random_state)
    self.estimators = [] # (estimator, estimator_p_trn)
    for _ in range(self.n_estimators):
      if self.n_estimators > 1:
        p_e = rng.dirichlet(np.ones(self.n_classes))
        i_e = draw_indices(
          y,
          p_e,
          len(X),
          min_samples_per_class = self.min_samples_per_class,
          random_state = self.random_state,
        )
        p_e = n_samples_per_class(y[i_e]) / len(i_e) # correct
        assert len(np.setxor1d(np.unique(y), np.unique(y[i_e]))) == 0
        estimator = clone(self.base_estimator).fit(X[i_e], y[i_e])
        self.estimators.append((estimator, p_e))
      else:
        self.estimators.append((clone(self.base_estimator).fit(X, y), self.p_trn))
    return self
  def quantify(self, X):
    return self.predict(X)
  def predict(self, X):
    pXY = []
    for estimator, estimator_p_trn in self.estimators:
      pXY_i = estimator.predict_proba(X) / estimator_p_trn # P(X|Y)
      pXY.append(pXY_i / pXY_i.sum(axis=1, keepdims=True))
    pXY = np.concatenate(pXY) # concatenate along the dimension 0
    best_result = ( # tuple of loss and result
      np.inf,
      Result(np.ones(self.n_classes) / self.n_classes, 0, "unsolved")
    )
    rng = np.random.RandomState(self.random_state)
    dims_0 = np.arange(self.n_classes if self.multistart else 1)
    for dim_0 in dims_0:
      def fun(x):
        p = _jnp_softmax(x, dim_0)
        return -jnp.log(pXY @ p).mean() + self.tau * jnp.sum((p[1:] - p[:-1])**2) / 2
      jac = jax.grad(fun)
      hess = jax.jacfwd(jac) # forward-mode AD
      x0 = _rand_x0(rng, self.n_classes) # random starting point
      state = _CallbackState(x0)
      try:
        opt = minimize(
          fun, # JAX function l -> loss
          x0,
          jac = _check_derivative(jac, "jac"),
          hess = _check_derivative(hess, "hess"),
          method = self.solver,
          options = self.solver_options,
          callback = state.callback()
        )
      except (DerivativeError, ValueError):
        traceback.print_exc()
        opt = state.get_state()
      if opt.fun < best_result[0]:
        if dim_0 > 0:
          print(f"Improved prediction from {best_result[0]} to {opt.fun} (dim_0={dim_0})")
        best_result = (opt.fun, Result(_np_softmax(opt.x, dim_0), opt.nit, opt.message))
    return best_result[1]
