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
from sklearn.neural_network import MLPClassifier
from time import time
from . import MyGridSearchQ
from ..methods import KDEyMLQP, EMaxL
from ..utils import (load_lequa2024, evaluate_model, create_submission, mean_macro_normalized_match_distance, 
                     mean_normalized_match_distance)
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
        task
        ):
    """A single trial of lequa2022.main()"""
    np.random.seed(seed)
    print(
        f"INI [{i_trial+1:02d}/{n_trials:02d}]:",
        datetime.now().strftime('%H:%M:%S'),
        f"{method_name} / {data_name} starting"
    )

    # use tau_1 parameter for task T3
    if 'tau_0' in param_grid.keys() and task == 'T3': 
        param_grid['tau_1'] = param_grid['tau_0']
        param_grid.pop('tau_0')

    error_metric = 'mrae'
    if task == 'T3':
        error_metric = mean_normalized_match_distance

    # load the data
    if data_name == "lequa2024_val":
        X_trn, y_trn, val_gen, tst_gen = load_lequa2024(task=task) # like T1B from 2022
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
    quapy_method = MyGridSearchQ(
        model = method,
        param_grid = param_grid,
        protocol = val_gen,
        error = error_metric, # "mrae",
        extra_metrics = [
            "mean_logodds_uniformness_ratio_score",
            "mean_probability_uniformness_ratio_score",
        ],
        refit = False,
        n_jobs = n_jobs,
        raise_errors = True,
        verbose = True,
    ).fit(trn_data)
    val_results = quapy_method.param_scores_df_
    val_results["method"] = method_name
    val_results["data"] = data_name
    val_results["task"] = task
    print(
        f"VAL [{i_trial+1:02d}/{n_trials:02d}]:",
        datetime.now().strftime('%H:%M:%S'),
        f"{method_name} validated RAE={quapy_method.best_score_:.4f}",
        f"{quapy_method.best_params_}",
    )

    #_, _, val_gen, _ = load_lequa2024(task=task) # like T1B from 2022

    evaluate_model(quapy_method.best_model(), tst_gen, task, f"val_predictions_{task}.csv")

    # create submission file
    #create_submission(quapy_method.best_model(), tst_gen, f"{task}_submission_{method_name}_{quapy_method.best_score_:.4f}.txt")

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
    clf = LogisticRegression(max_iter=3000, tol=1e-6, random_state=seed)
    clf_MLP = MLPClassifier(max_iter=3000, tol=1e-6, random_state=seed)
    clf_grid = lambda prefix: {
        f"{prefix}__C": np.linspace(0.1, 0.35, 15),  # T1
        #f"{prefix}__C": np.linspace(0.35, 0.5, 15), # T2
        #f"{prefix}__C": np.concatenate([np.linspace(70, 270, 15), np.array([750, 800, 850])]), # T3
        #f"{prefix}__C": np.linspace(0.35, 0.75, 15), # T4
        f"{prefix}__class_weight": [None, 'balanced'],
    }
    clf_grid_mlp = lambda prefix: {
        f"{prefix}__hidden_layer_sizes" : [[320], [420], [512], [700]],
        f"{prefix}__activation" : ['tanh'],
        f"{prefix}__alpha" : [1e-3, 1e-2, 1e-1, 1e1], # L2-Reg-Term
        f"{prefix}__learning_rate_init" : [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        f"{prefix}__learning_rate" : ['constant'],
        f"{prefix}__solver" : ['adam', 'sgd'],
    }
    # mrae: 0.10913
    test_grid_t4 = {
        'tau_0': [1e-05],
        'n_estimators': [1],
        'base_estimator__C': [0.4357142857142857],
        'base_estimator__class_weight': [None]
    }
    # NMD: 0.06585
    test_grid_t3 = {
        'base_estimator__hidden_layer_sizes': [320],
        'base_estimator__activation': ['tanh'],
        'base_estimator__alpha': [1e-06],
        'base_estimator__learning_rate_init': [0.001],
        'base_estimator__learning_rate': ['constant'],
        'base_estimator__solver': ['adam'],
        'n_estimators': [1],
        'tau_1': [0.001]
    }
    # mrae: 0.103015
    test_grid_t2 = {
        'base_estimator__hidden_layer_sizes': [512],
        'base_estimator__activation': ['tanh'],
        'base_estimator__alpha': [0.1],
        'base_estimator__learning_rate_init': [1e-05],
        'base_estimator__learning_rate': ['constant'],
        'base_estimator__solver': ['adam']
    }
    # mrae: 0.10853
    test_grid_t1 = {
        'base_estimator__hidden_layer_sizes': [512],
        'base_estimator__activation': ['tanh'],
        'base_estimator__alpha': [0.1],
        'base_estimator__learning_rate_init': [1e-3],
        'base_estimator__learning_rate': ['constant'],
        'base_estimator__solver': ['sgd']
    }
    q_grid = {
        "tau_0": [0, 1e-5, 1e-3],
        "n_estimators" : [1],
    }
    if is_test_run: # use a minimal testing configuration
        clf = LogisticRegression(max_iter=3, random_state=seed)
        clf_grid = lambda prefix: {
            f"{prefix}__C": [1.0],
        }
        q_grid = {
            "tau_1": [1e1, 0],
        }
    methods = [ # (method_name, method, param_grid)
        # ("SLD", qp.method.aggregative.EMQ(clf), clf_grid("classifier")),
        #(
        #    "EMaxL",
        #    EMaxL(clf, n_estimators=1, random_state=seed),
        #    q_grid | clf_grid("base_estimator")
        #),
        (
            "EMaxL_MLP",
            EMaxL(clf_MLP, n_estimators=1, random_state=seed),
            clf_grid_mlp("base_estimator")
        ),
	    #(
	    #    "EMaxL",
	    #    EMaxL(clf, n_estimators=1, random_state=seed),
	    #    test_grid_t4
	    #)
    ]

    #tasks = ['T1', 'T2', 'T3', 'T4']
    tasks = ['T2']

    # iterate over all methods and data sets
    data_names = ["lequa2024_val"] # ["lequa2024_val", "lequa2022_val", "lequa2022_tst"]
    n_trials = len(methods) * len(data_names) * len(tasks)
    print(f"Starting {n_trials} trials")
    val_results = []
    for i_trial, (method, data_name, task) in enumerate(itertools.product(methods, data_names, tasks)):
        val_results.append(trial(
            i_trial,
            *method, # = (method_name, method, param_grid)
            data_name,
            n_jobs,
            seed,
            n_trials,
            is_test_run,
            task,
        ))
    val_df = pd.concat(val_results).reset_index(drop=True)
    val_df.to_csv(val_path) # store the results
    print(f"{val_df.shape[0]} validation results stored at {val_path}")

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
