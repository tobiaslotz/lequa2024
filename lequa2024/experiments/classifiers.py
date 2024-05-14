import argparse
import itertools
import numpy as np
import quapy as qp
import os
import pandas as pd
from datetime import datetime
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from ..utils import load_lequa2024
import warnings
warnings.filterwarnings("ignore")

def trial(
        i_trial,
        classifier_name,
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
        f"{classifier_name} / {data_name} starting"
    )

    # load the data
    if data_name == "lequa2024":
        X_trn, y_trn, _, _ = load_lequa2024(task="T2") # like T1B from 2022
        trn_data = qp.data.base.LabelledCollection(X_trn, y_trn)
    elif data_name == "lequa2022":
        trn_data, _, _ = qp.datasets.fetch_lequa2022(task="T1B")
    else:
        raise ValueError(f"Unknown data_name={data_name}")

    # configure the grid search
    if classifier_name == "lr":
        cv = GridSearchCV(
            LogisticRegression(random_state=seed),
            {
                "C": np.logspace(-3, 0, 10),
            },
            n_jobs = n_jobs,
            refit = False,
        )
    elif classifier_name == "mlp":
        cv = GridSearchCV(
            MLPClassifier(random_state=seed, max_iter=2000, early_stopping=True),
            {
                "activation": ["tanh"],
                "hidden_layer_sizes": [
                    (256, 128, 64),
                    (256, 128),
                    (128, 64),
                    (256,)
                ],
                "learning_rate_init": np.logspace(-3, -4, 5),
                "alpha": np.logspace(1, -5, 4),
            },
            n_jobs = n_jobs,
            refit = False,
        )
    else:
        raise ValueError(f"Unknown classifier_name={classifier_name}")

    # set up a minimal testing configuration
    if is_test_run:
        trn_data = trn_data.split_stratified(3000, random_state=seed)[0] # subsample
        if classifier_name == "lr":
            cv = GridSearchCV(
                LogisticRegression(random_state=seed, max_iter=3),
                {
                    "C": np.logspace(-3, 0, 2)
                },
                n_jobs = n_jobs,
                refit = False,
                verbose = 3, # score and time for each fold and parameter setup
            )
        elif classifier_name == "mlp":
            cv = GridSearchCV(
                MLPClassifier(random_state=seed, max_iter=3),
                {
                    "activation": ["tanh"],
                    "hidden_layer_sizes": [(64,)],
                    "learning_rate_init": np.logspace(-2, -4, 2),
                },
                n_jobs = n_jobs,
                refit = False,
            )
        else:
            raise ValueError(f"Unknown classifier_name={classifier_name}")

    cv.fit(*trn_data.Xy)
    val_results = pd.DataFrame(cv.cv_results_)[["params", "mean_test_score", "std_test_score", "mean_fit_time"]]
    val_results["classifier"] = classifier_name
    val_results["data"] = data_name
    print(
        f"VAL [{i_trial+1:02d}/{n_trials:02d}]:",
        datetime.now().strftime('%H:%M:%S'),
        f"{classifier_name} / {data_name} validated Acc={cv.best_score_:.4f}",
        f"{cv.best_params_}",
    )
    return val_results

def main(
        val_path,
        n_jobs = 1,
        seed = 867,
        is_test_run = False,
    ):
    print(f"Starting a classifiers experiment to produce {val_path} with seed {seed}")
    if is_test_run:
        print("WARNING: this is a test run; results are not meaningful")
    if len(os.path.dirname(val_path)) > 0: # ensure that the directory exists
        os.makedirs(os.path.dirname(val_path), exist_ok=True)
    np.random.seed(seed)

    # iterate over all data sets
    classifier_names = ["mlp"] # ["lr", "mlp"]
    data_names = ["lequa2024"] # ["lequa2022", "lequa2024"]
    n_trials = len(classifier_names) * len(data_names)
    print(f"Starting {n_trials} trials")
    val_results = []
    for i_trial, (classifier_name, data_name) in enumerate(itertools.product(classifier_names, data_names)):
        val_results.append(trial(
            i_trial,
            classifier_name,
            data_name,
            n_jobs,
            seed,
            n_trials,
            is_test_run,
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
