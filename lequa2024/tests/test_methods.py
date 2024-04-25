import jax.numpy as jnp
import numpy as np
import quapy as qp
import qunfold
from lequa2024.methods import EMaxL, KDEyMLQP
from quapy.model_selection import GridSearchQ
from qunfold.sklearn import CVClassifier
from sklearn.linear_model import LogisticRegression
from unittest import TestCase

class TestMethods(TestCase):
  def test_methods(self):
    np.random.seed(25)
    qp.environ["_R_SEED"] = 25
    qp.environ["SAMPLE_SIZE"] = 1000

    # load the data
    trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")
    val_gen.true_prevs.df = val_gen.true_prevs.df[:7] # use only 7 validation samples

    # configure the quantification methods
    clf = LogisticRegression(C=0.01)
    methods = [ # (method_name, method, param_grid)
        ("SLD ...........", qp.method.aggregative.EMQ(clf)),
        ("EMaxL (ours) ..", EMaxL(clf, n_estimators=30, random_state=25)),
        ("MaxL (ours) ...", EMaxL(clf, n_estimators=1, random_state=25)),
        ("KDEy (original)", KDEyMLQP(clf, random_state=25)),
    ]

    # evaluate
    for method_name, method in methods:
        method.fit(trn_data)
        errors = qp.evaluation.evaluate( # errors of all predictions
            method,
            protocol = val_gen,
            error_metric = "rae",
        )
        avg = errors.mean()
        errors = ", ".join([f"{error:.4f}" for error in errors]) # format
        print(f"{method_name} validates RAE=({errors}), avg={avg:.4f}")
