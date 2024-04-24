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
        tst_gen,
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
        quapy_method = qp.model_selection.GridSearchQ(
            model = method,
            param_grid = param_grid,
            protocol = val_gen,
            error = "m" + error_metric, # ae -> mae, rae -> mrae
            refit = False,
            n_jobs = n_jobs,
            raise_errors = True,
            verbose = True,
        ).fit(trn_data)
        parameters = quapy_method.best_params_
        val_error = quapy_method.best_score_
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
        print(
            f"Skipping validation of {method_name} due to fixed hyper-parameters."
        )

    # evaluate the method on the test samples and return the result
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
    return {
        "method": method_name,
        "error_metric": error_metric,
        "error": error,
        "error_std": error_std,
        "prediction_time": prediction_time,
        "val_error": val_error,
        "parameters": str(parameters),
    }

def main(
        output_path,
        n_jobs = 1,
        seed = 867,
        is_full_run = False,
        is_test_run = False,
    ):
    print(f"Starting a lequa2022 experiment to produce {output_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    if len(os.path.dirname(output_path)) > 0: # ensure that the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
        "transformer__classifier__estimator__C": [1e-3, 1e-2, 1e-1, 1e0, 1e1],
        "transformer__classifier__estimator__class_weight": ["balanced", None],
    }
    qp_clf = clf.estimator
    qp_clf_grid = {
        "classifier__C": clf_grid["transformer__classifier__estimator__C"],
        "classifier__class_weight": clf_grid["transformer__classifier__estimator__class_weight"],
    }
    methods = [ # (method_name, method, param_grid)
        ("PACC", QuaPyWrapper(PACC(clf, seed=seed)), clf_grid),
        ("SLD", qp.method.aggregative.EMQ(qp_clf), qp_clf_grid),
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
            ("PACC", QuaPyWrapper(PACC(clf, seed=seed)), clf_grid),
            ("SLD", qp.method.aggregative.EMQ(qp_clf), qp_clf_grid),
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
        tst_gen = tst_gen,
        n_jobs = n_jobs,
        seed = seed,
        n_trials = len(trials),
    )
    print(
        f"Starting {len(trials)} trials",
        f"with {len(val_gen.true_prevs.df)} validation",
        f"and {len(tst_gen.true_prevs.df)} testing samples",
    )
    results = []
    for trial_config in trials:
        results.append(configured_trial(*trial_config))
    df = pd.DataFrame(results)
    df.to_csv(output_path) # store the results
    print(f"{df.shape[0]} results succesfully stored at {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', type=str, help='path of an output *.csv file')
    parser.add_argument('--n_jobs', type=int, default=1, metavar='N',
                        help='number of concurrent jobs or 0 for all processors (default: 1)')
    parser.add_argument('--seed', type=int, default=876, metavar='N',
                        help='random number generator seed (default: 876)')
    parser.add_argument("--is_full_run", action="store_true",
                        help="whether to use all 1000 validation and 5000 testing samples")
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.output_path,
        args.n_jobs,
        args.seed,
        args.is_full_run,
        args.is_test_run,
    )
