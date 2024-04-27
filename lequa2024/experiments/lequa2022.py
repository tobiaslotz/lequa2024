import argparse
import itertools
import numpy as np
import os
import pandas as pd
import quapy as qp
from datetime import datetime
from functools import partial
from qunfold import PACC
from qunfold.quapy import QuaPyWrapper
from qunfold.sklearn import CVClassifier
from sklearn.linear_model import LogisticRegression
from time import time
from . import MyGridSearchQ
from ..methods import KDEyMLQP, EMaxL
import warnings
warnings.filterwarnings("ignore")

def trial(
        i_method,
        method_name,
        method,
        param_grid,
        error_metric,
        trn_data,
        val_gen,
        tst_gen, # set to None to omit testing
        n_jobs,
        seed,
        n_trials,
        ):
    """A single trial of lequa2022.main()"""
    np.random.seed(seed)
    print(
        f"INI [{i_method+1:02d}/{n_trials:02d}]:",
        datetime.now().strftime('%H:%M:%S'),
        f"{method_name} / {error_metric} starting"
    )

    # configure and train the method; select the best hyper-parameters
    if param_grid is not None:
        quapy_method = MyGridSearchQ(
            model = method,
            param_grid = param_grid,
            protocol = val_gen,
            error = "m" + error_metric, # ae -> mae, rae -> mrae
            extra_metrics = [
                "mean_logodds_uniformness_ratio_score",
                "mean_probability_uniformness_ratio_score",
            ],
            refit = False,
            n_jobs = n_jobs,
            raise_errors = True,
            verbose = True,
        ).fit(trn_data)
        parameters = quapy_method.best_params_
        val_error = quapy_method.best_score_
        val_results = quapy_method.param_scores_df_
        val_results["method"] = method_name
        quapy_method = quapy_method.best_model_
        print(
            f"VAL [{i_method+1:02d}/{n_trials:02d}]:",
            datetime.now().strftime('%H:%M:%S'),
            f"{method_name} validated {error_metric}={val_error:.4f} {parameters}"
        )
    else:
        quapy_method = method.fit(trn_data)
        parameters = None
        val_error = -1
        val_results = pd.DataFrame()
        print(
            f"VAL [{i_method+1:02d}/{n_trials:02d}]:",
            f"Skipping validation of {method_name} due to fixed hyper-parameters"
        )

    # evaluate the method on the test samples and return the result
    if tst_gen is not None:
        t_0 = time()
        errors = qp.evaluation.evaluate( # errors of all predictions
            quapy_method,
            protocol = tst_gen,
            error_metric = error_metric,
            verbose = True,
        )
        prediction_time = (time() - t_0) / len(errors) # average prediction_time
        error = errors.mean()
        error_std = errors.std()
        print(
            f"TST [{i_method+1:02d}/{n_trials:02d}]:",
            datetime.now().strftime('%H:%M:%S'),
            f"{method_name} tested {error_metric}={error:.4f}+-{error_std:.4f}"
        )
        tst_result = {
            "method": method_name,
            "error_metric": error_metric,
            "error": error,
            "error_std": error_std,
            "prediction_time": prediction_time,
            "val_error": val_error,
            "parameters": str(parameters),
        }
    else:
        tst_result = {}
        print(f"TST [{i_method+1:02d}/{n_trials:02d}]: Skipping testing of {method_name}")
    return val_results, tst_result

def main(
        val_path,
        tst_path,
        omit_testing = False,
        n_jobs = 1,
        seed = 867,
        is_full_run = False,
        is_test_run = False,
    ):
    print(f"Starting a lequa2022 experiment to produce {tst_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    if len(os.path.dirname(tst_path)) > 0: # ensure that the directory exists
        os.makedirs(os.path.dirname(tst_path), exist_ok=True)
    np.random.seed(seed)
    qp.environ["_R_SEED"] = seed
    qp.environ["SAMPLE_SIZE"] = 1000

    # configure the quantification methods
    clf = CVClassifier(
        LogisticRegression(),
        n_estimators = 10,
        random_state = seed,
    )
    clf_grid = {
        "transformer__classifier__estimator__C": np.logspace(-3, -1, 11),
    }
    qp_clf = clf.estimator
    qp_clf_grid = {
        "classifier__C": clf_grid["transformer__classifier__estimator__C"],
    }
    methods = [ # (method_name, method, param_grid)
        # ("PACC", QuaPyWrapper(PACC(clf, seed=seed)), clf_grid),
        # ("KDEy", KDEyMLQP(qp_clf, random_state=seed), {
        #     "bandwidth": np.linspace(0.01, 0.2, 20),
        #     "classifier__C": np.logspace(-3, 3, 7),
        #     "classifier__class_weight" : ["balanced", None],
        #     # **qp_clf_grid,
        # }),
        ("EMaxL", EMaxL(qp_clf, n_estimators=1, random_state=seed), {
            "base_estimator__C": clf_grid["transformer__classifier__estimator__C"],
            "tau": np.hstack([0, np.logspace(-7, -4, 4)])
        }),
    ]

    # load the data
    trn_data, val_gen, tst_gen = qp.datasets.fetch_lequa2022(task="T1B")

    if is_test_run: # use a minimal testing configuration
        clf.set_params(n_estimators = 3, estimator__max_iter = 3)
        clf_grid = {
            "transformer__classifier__estimator__C": [1e1],
        }
        qp_clf = clf.estimator
        qp_clf_grid = {
            "classifier__C": clf_grid["transformer__classifier__estimator__C"],
        }
        methods = [ # (method_name, method, param_grid)
            # ("PACC", QuaPyWrapper(PACC(clf, seed=seed)), clf_grid),
            # ("KDEy", KDEyMLQP(qp_clf, random_state=seed), {
            #     "bandwidth": np.linspace(0.01, 0.2, 2),
            # }),
            ("EMaxL", EMaxL(qp_clf, n_estimators=1, random_state=seed), {
                "base_estimator__C": clf_grid["transformer__classifier__estimator__C"],
                "tau": [0, 0.1]
            }),
        ]
        trn_data = trn_data.split_stratified(3000, random_state=seed)[0] # subsample
        val_gen.true_prevs.df = val_gen.true_prevs.df[:3] # use only 3 validation samples
        tst_gen.true_prevs.df = tst_gen.true_prevs.df[:3] # use only 3 testing samples
    elif not is_full_run:
        val_gen.true_prevs.df = val_gen.true_prevs.df[:100] # use only 100 validation samples
        tst_gen.true_prevs.df = tst_gen.true_prevs.df[:500] # use only 500 testing samples

    # iterate over all methods
    error_metrics = ["rae"] # "ae"
    trials = [ # (i_method, method_name, method, param_grid, error_metric)
        (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][1])
        for x in enumerate(itertools.product(methods, error_metrics))
    ]
    configured_trial = partial(
        trial,
        trn_data = trn_data,
        val_gen = val_gen,
        tst_gen = tst_gen if not omit_testing else None,
        n_jobs = n_jobs,
        seed = seed,
        n_trials = len(trials),
    )
    print(
        f"Starting {len(trials)} trials",
        f"with {len(val_gen.true_prevs.df)} validation",
        f"and {len(tst_gen.true_prevs.df)} testing samples",
    )
    val_results = []
    tst_results = []
    for trial_config in trials:
        trial_val_results, trial_tst_results = configured_trial(*trial_config)
        val_results.append(trial_val_results)
        tst_results.append(trial_tst_results)
    val_df = pd.concat(val_results).reset_index(drop=True)
    val_df.to_csv(val_path) # store the results
    print(f"{val_df.shape[0]} validation results stored at {val_path}")
    if not omit_testing:
        tst_df = pd.DataFrame(tst_results)
        tst_df.to_csv(tst_path)
        print(f"{tst_df.shape[0]} testing results stored at {tst_path}")
    else:
        print("Not writing any testing results")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('val_path', type=str, help='path of a validation output *.csv file')
    parser.add_argument('tst_path', type=str, help='path of a testing output *.csv file')
    parser.add_argument("--omit_testing", action="store_true",
                        help="whether to omit testing")
    parser.add_argument('--n_jobs', type=int, default=1, metavar='N',
                        help='number of concurrent jobs or 0 for all processors (default: 1)')
    parser.add_argument('--seed', type=int, default=876, metavar='N',
                        help='random number generator seed (default: 876)')
    parser.add_argument("--is_full_run", action="store_true",
                        help="whether to use all 1000 validation and 5000 testing samples")
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.val_path,
        args.tst_path,
        args.omit_testing,
        args.n_jobs,
        args.seed,
        args.is_full_run,
        args.is_test_run,
    )
