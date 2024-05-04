from .data import load_vector_documents, ResultSubmission, gen_load_samples
from .evaluate import evaluate_submission
from .constants import SAMPLE_SIZE
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import zipfile
from quapy.util import download_file_if_not_exists

metric_map = {'T1' : ['ae', 'rae'],
              'T2' : ['ae', 'rae'],
              'T3' : ['macro-nmd', 'nmd'],
              'T4' : ['ae', 'rae']}

def load_lequa2024(task='T1', data_dir=None):
    task = task.lower()
    if task != 't1' and task != 't2' and task != 't3' and task != 't4':
        raise ValueError(f'Invalid task specification {task.upper()}. T1-T4 are supported')
    task = task.upper()
    if data_dir is None:
        data_dir = os.path.join(str(Path.home()), 'lequa2024_data/')
    os.makedirs(data_dir, exist_ok=True)

    URL_TRAIN_DEV = f'https://zenodo.org/records/11091067/files/{task}.train_dev.zip'
    URL_TEST = f'https://zenodo.org/records/11091067/files/{task}.test.zip'

    def download_data(url, is_train=True):
        train_or_test = "test"
        if is_train:
            train_or_test = "train"
        tmp_path = os.path.join(data_dir, f'{train_or_test}/{task}_tmp.zip')
        download_file_if_not_exists(url, tmp_path)
        with zipfile.ZipFile(tmp_path) as file:
            file.extractall(os.path.join(data_dir, train_or_test))
        os.remove(tmp_path)

    if task == 'T3':  
        train_dir = os.path.join(data_dir, f'train/{task}/public/training_samples/')
        if not os.path.exists(train_dir):
            download_data(URL_TRAIN_DEV, is_train=True)
        N_SAMPLE_FILES = 100
        X_train = np.zeros((N_SAMPLE_FILES, SAMPLE_SIZE[task], 256))
        y_train = np.zeros((N_SAMPLE_FILES, SAMPLE_SIZE[task]), dtype=np.int32)
        for i in range(N_SAMPLE_FILES):
            X_train[i], y_train[i] = load_vector_documents(os.path.join(train_dir, f'{i}.txt'))
    else:
        train_data_path = os.path.join(data_dir, f'train/{task}/public/training_data.txt')
        if not os.path.exists(train_data_path):
            download_data(URL_TRAIN_DEV, is_train=True)
        X_train, y_train = load_vector_documents(train_data_path)

    val_dir = os.path.join(data_dir, f'train/{task}/public/dev_samples')
    val_gt_path = os.path.join(data_dir, f'train/{task}/public/dev_prevalences.txt')
    val_gen = gen_load_samples(val_dir, val_gt_path)

    test_dir = os.path.join(data_dir, f'test/{task}/public/test_samples')
    if not os.path.exists(test_dir):
        download_data(URL_TEST, is_train=False)
    test_gen = gen_load_samples(test_dir)

    return X_train, y_train, val_gen, test_gen

def evaluate_model(model, protocol, task, pred_path=None):
    pred_prevs = ResultSubmission()
    true_prevs = ResultSubmission()
    print('Starting evaluation ...')
    for id, sample, gt in tqdm(protocol, total=1000):
        preds = model.predict(sample)
        pred_prevs.add(id, preds)
        true_prevs.add(id, gt)
    if pred_path is not None and isinstance(pred_path, str):
        pred_prevs.dump(pred_path)
    metrics = metric_map[task]
    errors = []
    print('\nCalculating metrics ...')
    for metric in metrics:
        eval = evaluate_submission(true_prevs, pred_prevs, SAMPLE_SIZE[task], metric, average=False)
        errors.append((metric, eval.mean(), eval.std()))
        print(f'm{metric}: {eval.mean():.5f} ~ {eval.std():.5f}')
    return errors
