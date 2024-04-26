import numpy as np
import quapy as qp
from lequa2024.experiments import (
    uniformness_ratio_score,
    mean_uniformness_ratio_score,
    my_evaluate,
    MyGridSearchQ,
)
from sklearn.linear_model import LogisticRegression
from unittest import TestCase

class TestExperiments(TestCase):
  def test_uniformness_ratio_score(self):
    np.random.seed(25)
    n_classes = 3
    n_prevs = 10
    p = np.random.dirichlet(np.ones(n_classes), size=n_prevs)
    score = uniformness_ratio_score(p, p)
    self.assertEqual(score.shape, (n_prevs,))
    np.testing.assert_equal(score, 1)
    p2 = np.random.dirichlet(np.ones(n_classes), size=n_prevs)
    score = uniformness_ratio_score(p, p)
    self.assertEqual(score.shape, (n_prevs,))
    self.assertEqual(score.mean(), mean_uniformness_ratio_score(p, p))

  def test_my_evaluate(self):
    np.random.seed(25)
    qp.environ["_R_SEED"] = 25
    qp.environ["SAMPLE_SIZE"] = 1000

    # load the data
    trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")
    trn_data, _ = trn_data.split_stratified(train_prop=3000) # use only 3k training samples
    val_gen.true_prevs.df = val_gen.true_prevs.df[:5] # use only 5 validation samples

    method = qp.method.aggregative.EMQ(LogisticRegression(C=0.01)).fit(trn_data)
    rae = qp.evaluation.evaluate(method, protocol=val_gen, error_metric="mrae")
    mae = qp.evaluation.evaluate(method, protocol=val_gen, error_metric="mae")
    errors = my_evaluate(method, protocol=val_gen, error_metric="mrae") # no extra_metrics
    self.assertEqual(errors[0], rae)
    errors = my_evaluate( # with extra_metrics
      method,
      protocol = val_gen,
      error_metric = "mrae",
      extra_metrics = [ "mae" ]
    )
    self.assertEqual(errors[0], rae)
    self.assertEqual(errors[1], mae)
    errors = my_evaluate( # with vector outputs
      method,
      protocol = val_gen,
      error_metric = "rae",
      extra_metrics = [ "ae" ]
    )
    np.testing.assert_equal(
      errors[0],
      qp.evaluation.evaluate(method, protocol=val_gen, error_metric="rae")
    )
    np.testing.assert_equal(
      errors[1],
      qp.evaluation.evaluate(method, protocol=val_gen, error_metric="ae")
    )
    errors = my_evaluate( # with mean_uniformness_ratio_score
      method,
      protocol = val_gen,
      error_metric = "mrae",
      extra_metrics = [ "mean_uniformness_ratio_score" ]
    )
    self.assertGreaterEqual(errors[1], 0)

  def test_MyGridSearchQ(self):
    np.random.seed(25)
    qp.environ["_R_SEED"] = 25
    qp.environ["SAMPLE_SIZE"] = 1000

    # load the data
    trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")
    trn_data, _ = trn_data.split_stratified(train_prop=3000) # use only 3k training samples
    val_gen.true_prevs.df = val_gen.true_prevs.df[:5] # use only 5 validation samples

    gs = MyGridSearchQ(
      qp.method.aggregative.EMQ(LogisticRegression()),
      param_grid = { "classifier__C": [ 1e-2, 1e-1 ]},
      protocol = val_gen,
      error = "mrae",
      extra_metrics = [ "mean_uniformness_ratio_score" ],
      refit = False,
      raise_errors = True,
      verbose = True,
    ).fit(trn_data)
    print(gs.param_scores_df_)
