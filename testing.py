from lequa2024.utils import load_lequa2024, evaluate_model
from qunfold import PACC
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    task = 'T1'
    seed = 876
        
    X_train, y_train, val_gen, test_gen = load_lequa2024(task=task)

    if task == 'T3':
        X_train = X_train.reshape((X_train.shape[0]*X_train.shape[1], 256))
        y_train = y_train.flatten()

    pacc = PACC(classifier=RandomForestClassifier(oob_score=True, random_state=seed)).fit(X_train, y_train)
    print('Method: PACC')
    errs_pacc = evaluate_model(pacc, val_gen, task=task)
