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
from time import time
from . import MyGridSearchQ
from ..methods import KDEyMLQP, EMaxL
from ..neural import MLPClassifier, SetTraining
from ..utils import load_lequa2024
import warnings
warnings.filterwarnings("ignore")

def trial(
        i_trial,
        method_name,
        method,
        param_grid,
        data_name,
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
        f"{method_name} / {data_name} starting"
    )

    # load the data
    if data_name == "lequa2024_val":
        X_trn, y_trn, val_gen, _ = load_lequa2024(task="T2") # like T1B from 2022
        trn_data = qp.data.base.LabelledCollection(X_trn, y_trn)
    elif data_name == "lequa2022_val":
        trn_data, val_gen, _ = qp.datasets.fetch_lequa2022(task="T1B")
    elif data_name == "lequa2022_tst":
        trn_data, _, val_gen = qp.datasets.fetch_lequa2022(task="T1B")
    else:
        raise ValueError(f"Unknown data_name={data_name}")
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
    method.classifier.set_training = SetTraining(val_X, val_p)
    cv = MyGridSearchQ(
        model = method,
        param_grid = param_grid,
        protocol = val_gen,
        error = "mrae",
        extra_metrics = [
            "mean_logodds_uniformness_ratio_score",
            "mean_probability_uniformness_ratio_score",
        ],
        store_progress = True,
        refit = False,
        n_jobs = n_jobs,
        raise_errors = True,
        verbose = True,
    ).fit(trn_data)
    val_results = cv.param_scores_df_
    val_results["method"] = method_name
    val_results["data"] = data_name
    print(
        f"VAL [{i_trial+1:02d}/{n_trials:02d}]:",
        datetime.now().strftime('%H:%M:%S'),
        f"{method_name} validated RAE={cv.best_score_:.4f}",
        f"{cv.best_params_}",
    )
    progress = cv.progress_
    progress["method"] = method_name
    progress["data"] = data_name
    return val_results, progress

def main(
        val_path,
        trn_path,
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
    clf = MLPClassifier(random_state=seed, verbose=True)
    clf_grid = lambda prefix: {
        f"{prefix}__n_features": [
            # (256, 256),
            # (128, 128),
            # (512,),
            (256,),
            # (128,),
        ],
        f"{prefix}__lr_init": [0.01], # np.logspace(-1, -3, 3),
        f"{prefix}__batch_size": [128],
        f"{prefix}__activation": ["tanh"], # , "sigmoid", "relu"
    }
    if is_test_run: # use a minimal testing configuration
        clf = MLPClassifier(
            random_state = seed,
            n_epochs = 3,
            n_epochs_between_val = 1,
            verbose = True
        )
        clf_grid = lambda prefix: {
            f"{prefix}__n_features": [(64,)],
            f"{prefix}__lr_init": [0.01],
            f"{prefix}__batch_size": [64],
            f"{prefix}__activation": ["tanh", "sigmoid", "relu"],
        }
    methods = [ # (method_name, method, param_grid)
        ("SLD", qp.method.aggregative.EMQ(clf), clf_grid("classifier")),
        # ("EMaxL", EMaxL(clf, n_estimators=1, random_state=seed), clf_grid("base_estimator")),
    ]

    # iterate over all methods and data sets
    data_names = ["lequa2024_val"] # ["lequa2024_val", "lequa2022_val", "lequa2022_tst"]
    n_trials = len(methods) * len(data_names)
    print(f"Starting {n_trials} trials")
    val_results = []
    trn_progress = []
    for i_trial, (method, data_name) in enumerate(itertools.product(methods, data_names)):
        trial_results, trial_progress = trial(
            i_trial,
            *method, # = (method_name, method, param_grid)
            data_name,
            n_jobs,
            seed,
            n_trials,
            is_test_run,
        )
        val_results.append(trial_results)
        trn_progress.append(trial_progress)
    val_df = pd.concat(val_results).reset_index(drop=True)
    val_df.to_csv(val_path) # store the results
    trn_df = pd.concat(trn_progress).reset_index(drop=True)
    trn_df.to_csv(trn_path)
    print(
        f"{val_df.shape[0]} validation results stored at {val_path};",
        f"training progress stored at {trn_path}",
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('val_path', type=str, help='path of a validation output *.csv file')
    parser.add_argument('trn_path', type=str, help='path of a training output *.csv file')
    parser.add_argument('--n_jobs', type=int, default=1, metavar='N',
                        help='number of concurrent jobs or 0 for all processors (default: 1)')
    parser.add_argument('--seed', type=int, default=876, metavar='N',
                        help='random number generator seed (default: 876)')
    parser.add_argument("--is_test_run", action="store_true")
    args = parser.parse_args()
    main(
        args.val_path,
        args.trn_path,
        args.n_jobs,
        args.seed,
        args.is_test_run,
    )
