import jax
import jax.numpy as jnp
import numpy as np
import traceback
from abc import abstractmethod
from scipy.optimize import minimize
from scipy.stats import scoreatpercentile
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_is_fitted, NotFittedError
from sklearn.model_selection import cross_val_predict
from quapy.method.base import BaseQuantifier
from qunfold.methods import Result
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
  def solve(self, X):
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
  def solve(self, X):
    np.random.RandomState(self.random_state)
    epsilon = 1e-10
    n_classes = len(self.mixture_components)
    test_densities = [np.exp(mc.score_samples(X)) for mc in self.mixture_components]
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
