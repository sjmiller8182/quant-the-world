
import warnings

import pandas as pd 
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt 
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    make_scorer,
    accuracy_score
    )


def cost_score(y, y_pred, fp_cost=25, fn_cost=125):
    '''
    '''
    # get the misclassifications
    misclass_idx = np.where(np.equal(y, y_pred) == False)[0]
    # get the false positives
    fp_idx = np.where(y_pred[misclass_idx] == 1)[0]
    # get the false negatives
    fn_idx = np.where(y_pred[misclass_idx] == 0)[0]
    # calc the misclassification cost
    misclassification_cost = fp_idx.size * fp_cost + fn_idx.size * fn_cost
    
    return misclassification_cost


warnings.filterwarnings('ignore')

# pd.options.display.max_columns = 100
random_state = 42

random_generator = np.random.RandomState(random_state)

cost_scorer = make_scorer(cost_score, greater_is_better=False)



data = pd.read_csv('./final_project.csv')

y = data['y']
X = data.drop(['y'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
print('X_train: ', X_train.shape,
      '\ny_train: ', y_train.shape,
      '\nX_test: ', X_test.shape,
      '\ny_test: ', y_test.shape)


# fix spelling error
X_test['x24'] = X_test['x24'].str.replace('euorpe', 'europe')
# remove  %
X_test['x32'] = pd.to_numeric(X_test['x32'].str.replace('%', ''))
# remove $
X_test['x37'] = pd.to_numeric(X_test['x37'].str.replace('$', ''))
# repeat process for training set
X_train['x24'] = X_train['x24'].str.replace('euorpe', 'europe')
X_train['x32'] = pd.to_numeric(X_train['x32'].str.replace('%', ''))
X_train['x37'] = pd.to_numeric(X_train['x37'].str.replace('$', ''))
# remake objects
objects = X_train.select_dtypes(['O'])
objects_test = X_test.select_dtypes(['O'])

# imputing with mode from training data
X_train['x24'].fillna('asia', inplace=True)
X_train['x29'].fillna('July', inplace=True)
X_train['x30'].fillna('wednesday', inplace=True)

X_test['x24'].fillna('asia', inplace=True)
X_test['x29'].fillna('July', inplace=True)
X_test['x30'].fillna('wednesday', inplace=True)


names = [i for i in list(objects.columns)]

le = LabelEncoder()
for i in names:
    le.fit(objects[i].astype(str))
    X_train[i] = le.transform(X_train[i])
    X_test[i] = le.transform(X_test[i])

KNNimp = KNNImputer(n_neighbors=3)
X_train = KNNimp.fit_transform(X_train)
X_test = KNNimp.transform(X_test)

# define the estimator
logistic = LogisticRegression()
# provide the parameters of the feature selection process
feature_selector = RFECV(logistic,
          step = 1,
          min_features_to_select= 1,
          cv = 5,
          n_jobs = -1)
feature_selector = feature_selector.fit(X_train, y_train)


X_train = feature_selector.transform(X_train)
X_test = feature_selector.transform(X_test)
print('X_train shape: ', X_train.shape, 
      '\nX_test shape: ', X_test.shape)





xgb_params = {
    'n_estimators': np.arange(100, 1000, 10, dtype='int'),
    'learning_rate': np.linspace(0.01, 1, num=1000, dtype='float'),
    'gamma':np.geomspace(0.001, 10, num=1000, dtype='float'),
    'max_depth':[d for d in range(1, 11)],
    'subsample':np.linspace(0.1, 1, num=100, dtype='float'),
    'colsample_bytree':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bynode':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'lambda': np.geomspace(0.001, 10, num=100, dtype='float'),
    'alpha': np.geomspace(0.001, 10, num=100, dtype='float')
}

xgb = XGBClassifier(booster='gbtree',
                    early_stopping_rounds=10,
                    random_state=random_state,
                    nthread=36)

xgb_search = RandomizedSearchCV(xgb, 
                                xgb_params,
                                random_state=random_state,
                                scoring=cost_scorer,
                                n_iter=1000,
                                cv=5,
                                verbose=0,
                                n_jobs=-1)

xgb_search.fit(X_train, y_train)

y_pred = xgb_search.best_estimator_.predict(X_train)

print('\n\n\nTraining Performance')

print('Best model Score:', -xgb_search.best_score_) # negate since 'greater_is_better=False'
print('Best model Accuracy:', accuracy_score(y_train, y_pred) )



y_pred = xgb_search.best_estimator_.predict(X_test)

test_cost = cost_score(y_test, y_pred)
test_acc = accuracy_score(y_test, y_pred)

print('\n\n\nTest Performance')

print('Best Model Test Cost', test_cost)
print('Best Model Test Accuracy', test_acc)

print('\n\n\nBest Parameters')
print(xgb_search.best_params_)
