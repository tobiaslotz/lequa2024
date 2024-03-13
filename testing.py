from utils import load_lequa2024, evaluate_model
from qunfold import KMM, PACC
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    task = 'T4'

    X, y = load_lequa2024(task=task)

    if task == 'T3':
        X = X.reshape((X.shape[0]*X.shape[1], 256))
        y = y.flatten()

    pacc = PACC(classifier=RandomForestClassifier(oob_score=True)).fit(X, y)
    kmm = KMM().fit(X, y)
    print('Method: PACC')
    errs_pacc = evaluate_model(pacc, task=task, result_path="results_pacc.txt")
    print('\n\nMethod: KMM')
    errs_kmm = evaluate_model(kmm, task=task, result_path="results_kmm.txt")