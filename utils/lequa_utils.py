from utils.data import load_vector_documents, ResultSubmission, gen_load_samples
from utils.evaluate import evaluate_submission
from utils.constants import SAMPLE_SIZE
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

    URL_TRAIN = f'https://zenodo.org/records/10654475/files/{task}.train_dev.zip'

    def download_data(url):
        tmp_path = os.path.join(data_dir, f'{task}_tmp.zip')
        download_file_if_not_exists(url, tmp_path)
        with zipfile.ZipFile(tmp_path) as file:
            file.extractall(data_dir)
        os.remove(tmp_path)

    if task == 'T3':  
        train_dir = os.path.join(data_dir, f'{task}/public/training_samples/')
        if not os.path.exists(train_dir):
            download_data(URL_TRAIN)
        N_SAMPLE_FILES = 100
        X = np.zeros((N_SAMPLE_FILES, SAMPLE_SIZE[task], 256))
        y = np.zeros((N_SAMPLE_FILES, SAMPLE_SIZE[task]), dtype=np.int32)
        for i in range(N_SAMPLE_FILES):
            X[i], y[i] = load_vector_documents(os.path.join(train_dir, f'{i}.txt'))
    else:
        data_path = os.path.join(data_dir, f'{task}/public/training_data.txt')
        if not os.path.exists(data_path):
            download_data(URL_TRAIN)
        X, y = load_vector_documents(data_path)
    return X, y

def evaluate_model(model, task='T1', result_path=None, data_dir=None):
    task = task.lower()
    if task != 't1' and task != 't2' and task != 't3' and task != 't4':
        raise ValueError(f'Invalid task specification {task.upper()}. T1-T4 are supported')
    task = task.upper()
    if data_dir is None:
        data_dir = os.path.join(str(Path.home()), 'lequa2024_data/')
    sample_dir = os.path.join(data_dir, f'{task}/public/dev_samples')
    ground_truth_path = os.path.join(data_dir, f'{task}/public/dev_prevalences.txt')
    true_prevs = ResultSubmission().load(ground_truth_path)
    result = ResultSubmission()
    print('Starting evaluation ...')
    for id, sample, _ in tqdm(gen_load_samples(sample_dir, ground_truth_path, return_id=True), total=1000):
        preds = model.predict(sample)
        result.add(id, preds)
    if result_path is not None and isinstance(result_path, str):
        result.dump(result_path)
    metrics = metric_map[task]
    errors = []
    print('\nCalculating metrics ...')
    for metric in metrics:
        eval = evaluate_submission(true_prevs, result, SAMPLE_SIZE[task], metric, average=False)
        errors.append((metric, eval.mean(), eval.std()))
        print(f'm{metric}: {eval.mean():.5f} ~ {eval.std():.5f}')
    return errors
