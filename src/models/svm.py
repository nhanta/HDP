
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, auc, recall_score, f1_score,roc_auc_score, roc_curve, RocCurveDisplay, PrecisionRecallDisplay, precision_recall_curve, make_scorer
from sklearn.svm import SVC
from joblib import dump, load
import pandas as pd
import numpy as np
import shap

def shap_importance_getter(clf, X, y):
    explainer = shap.Explainer(clf)
    shap_values = explainer.shap_values(X, y)[0]
    importance = np.abs(shap_values).mean(0)
    return importance

# Select features using decision tree model
def rfe_svc(X_train, y_train, X_test, y_test):
    clf = SVC(random_state=7)
    best_auc = list()
    features = []
    iddd = []
    gr = []

    for i in range(1, len(X_train[0])):
        rfe = RFE(clf, n_features_to_select=i, importance_getter = shap_importance_getter(X_train, y_train))
        rfe.fit_transform(X_train, y_train)

        # Reduce X to the selected features.
        X_train_reduce = rfe.transform(X_train)
        X_test_reduce = rfe.transform(X_test)

        # Decision tree model
        roc_auc, grid = train_svc(X_train_reduce, y_train, X_test_reduce, y_test)
        print("Number of Selected Features", i, "AUC", roc_auc)
        features.append(rfe.support_)
        best_auc.append(roc_auc)
        gr.append(grid)
        iddd.append(i)

    print("The best AUC of Decision Tree: ", max(best_auc))
    idd = np.argmax(best_auc)
    print("Number of Selected Features is: ", iddd[idd])
    ft = features[idd] 

    dump(gr[idd], "svc.joblib")
    indice = [i for i, x in enumerate(ft) if x]
    
    pd.DataFrame({'features':indice}).to_csv('svc_features.csv')


def train_svc(X_train, y_train, X_test, y_test):

    # Create svm cross validation

    grid = {'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto', 0.001, 0.0001], \
            'kernel': ['rbf']}
    # 'rbf', 'sigmoid'
    clf = SVC(random_state=7)
    svc_grid = GridSearchCV(clf, grid, scoring='roc_auc', cv=5, n_jobs = 70)
    
    # Train the regressor
    svc_grid.fit(X_train, y_train)
    # Make predictions using the optimised parameters
    svc_pred = svc_grid.predict(X_test)
    
    roc_auc = round(roc_auc_score (y_test, svc_pred), 3)

    return (roc_auc, svc_grid)