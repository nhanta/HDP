
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, \
                            confusion_matrix, precision_score, auc, \
                            recall_score, f1_score,roc_auc_score, roc_curve, \
                            RocCurveDisplay, PrecisionRecallDisplay, \
                            precision_recall_curve, matthews_corrcoef, make_scorer

# Select features using decision tree model
def rfe_xgboost(X_train, y_train, X_test, y_test):
    
    estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=70,
    seed=42
    )
    best_auc = list()
    features = []
    iddd = []
    gr = []

    for i in range(1, len(X_train[0])):
        rfe = RFE(estimator, n_features_to_select=i)
        rfe.fit_transform(X_train, y_train)
        # Reduce X to the selected features.
        X_train_reduce = rfe.transform(X_train)
        X_test_reduce = rfe.transform(X_test)
        # Decision tree model
        roc_auc, grid_search = train_xgboost (X_train_reduce, y_train, X_test_reduce, y_test)
    
        features.append(rfe.support_)
        best_auc.append(roc_auc)
        gr.append(grid_search)
        iddd.append(i)

    print("The best AUC of Decision Tree: ", max(best_auc))
    idd = np.argmax(best_auc)
    print("Number of Selected Features is: ", iddd[idd])
    ft = features[idd] 

    dump(gr[idd], "xgboost.joblib")
    indice = [i for i, x in enumerate(ft) if x]
    
    pd.DataFrame({'features':indice}).to_csv('xgboost_features.csv')



def train_xgboost (X_train, y_train, X_test, y_test):

    # Create XGBoost cross validation
    estimator = XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
    )
    parameters = {
    'max_depth': range (2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
    }
    
    grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'roc_auc',
    n_jobs = 70,
    cv = 10,
    verbose=True
)
    
    # Train the regressor
    grid_search.fit(X_train, y_train)
    # Make predictions using the optimised parameters
    pred = grid_search.predict(X_test)

    roc_auc = round(roc_auc_score (y_test, pred), 3)

    return (roc_auc, grid_search)


    

    
    
