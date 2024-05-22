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
from ..neural_pcc import NeuralPCC, SetTraining
from ..utils import load_lequa2024
import warnings
warnings.filterwarnings("ignore")

def trial(
        i_trial,
        method_name,
        method,
        param_grid,
        n_jobs,
        seed,
        n_trials,
        is_test_run,
        ):
    """A single trial of lequa2022.main()"""
    np.random.seed(seed)
    print(
        f"INI [{i_trial+1:02d}/{n_trials:02d}]:",
        datetime.now().strftime('%H:%M:%S'),
        f"{method_name} starting"
    )

    # load the data
    X_trn, y_trn, val_gen, _ = load_lequa2024(task="T4")
    trn_data = qp.data.base.LabelledCollection(X_trn, y_trn)
    if is_test_run: # use a minimal testing configuration
        trn_data = trn_data.split_stratified(3000, random_state=seed)[0] # subsample
        val_gen.true_prevs.df = val_gen.true_prevs.df[:3] # use only 3 validation samples

    # configure and validate the method with all hyper-parameters
    print("Storing all validaton data for the set-based training")
    val_X = []
    val_p = []
    for val_gen_X, val_gen_p in val_gen():
        val_X.append(val_gen_X)
        val_p.append(val_gen_p)
    val_X = np.array(val_X) # concatenate along a newly introduced first dimension
    val_p = np.array(val_p)
    method.classifier.set_training = SetTraining(val_X, val_p) # TODO split
    cv = MyGridSearchQ(
        model = method,
        param_grid = param_grid,
        protocol = val_gen,
        error = "mrae",
        extra_metrics = [
            "mean_logodds_uniformness_ratio_score",
            "mean_probability_uniformness_ratio_score",
        ],
        store_progress = False,
        refit = False,
        n_jobs = n_jobs,
        raise_errors = True,
        verbose = True,
    ).fit(trn_data)
    val_results = cv.param_scores_df_
    val_results["method"] = method_name
    print(
        f"VAL [{i_trial+1:02d}/{n_trials:02d}]:",
        datetime.now().strftime('%H:%M:%S'),
        f"{method_name} validated RAE={cv.best_score_:.4f}",
        f"{cv.best_params_}",
    )
    return val_results

def main(
        val_path,
        n_jobs = 1,
        seed = 867,
        is_test_run = False,
    ):
    print(f"Starting a lequa2022 experiment to produce {val_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    if len(os.path.dirname(val_path)) > 0: # ensure that the directory exists
        os.makedirs(os.path.dirname(val_path), exist_ok=True)
    np.random.seed(seed)
    qp.environ["_R_SEED"] = seed
    qp.environ["SAMPLE_SIZE"] = 1000

    # configure the quantification methods
    clf_grid = {
        "classifier__C": np.logspace(4, 5, 2),
        "classifier__set_frac": [1], # np.linspace(.1, .9, 3),
        # "classifier__class_weight": ["balanced"], # [None, "balanced"]
    }
    if is_test_run: # use a minimal testing configuration
        clf = NeuralPCC(
            random_state = seed,
            max_iter = 3,
            verbose = True
        )
        clf_grid = {
            "classifier__C": [0.1],
            "classifier__set_frac": [1],
            # "classifier__class_weight": ["balanced"],
        }
    methods = [ # (method_name, method, param_grid)
        # ("sklearn", qp.method.aggregative.PCC(
        #     LogisticRegression(max_iter=3000, random_state=seed)
        # ), clf_grid),
        ("NeuralPCC", qp.method.aggregative.PCC(
            NeuralPCC(max_iter=3000, random_state=seed, verbose=True)
        ), clf_grid),
    ]

    # iterate over all methods and data sets
    print(f"Starting {len(methods)} trials")
    val_results = []
    for i_trial, method in enumerate(methods):
        trial_results = trial(
            i_trial,
            *method, # = (method_name, method, param_grid)
            n_jobs,
            seed,
            len(methods),
            is_test_run,
        )
        val_results.append(trial_results)
    val_df = pd.concat(val_results).reset_index(drop=True)
    val_df.to_csv(val_path) # store the results
    print(
        f"{val_df.shape[0]} validation results stored at {val_path}",
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('val_path', type=str, help='path of a validation output *.csv file')
    parser.add_argument('--n_jobs', type=int, default=1, metavar='N',
                        help='number of concurrent jobs or 0 for all processors (default: 1)')
    parser.add_argument('--seed', type=int, default=876, metavar='N',
                        help='random number generator seed (default: 876)')
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.val_path,
        args.n_jobs,
        args.seed,
        args.is_test_run,
    )
