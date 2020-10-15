import time
import pprint
import os
import sys

import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA 
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    make_scorer)
from sklearn.model_selection import (
    train_test_split, 
    RandomizedSearchCV, 
    cross_validate)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import xgboost as xgb

log_loss_scorer = make_scorer(log_loss, greater_is_better=False)
accuracy_scorer = make_scorer(accuracy_score)

random_state = np.random.RandomState(42)

# get the data
data = pd.read_csv('./data/case_8.csv')
# put the target in another variable
target = data.target
# drop off ID and target
data = data.drop(['ID', 'target'], axis=1)

# count the number of members in each categorigal feature
obj_columns = list(data.select_dtypes(include='object'))

categories = list()
for col in obj_columns:
    categories.append(
        len(np.unique(data[col]))
    )

for col, cat in zip(obj_columns, categories):
    print(col, cat)
    
# convert cats to one-hots
obj_columns = list(data.drop(['v22'], axis=1).select_dtypes(include='object'))
onehots = pd.get_dummies(data[obj_columns])

# label encode v22
data['v22'] = LabelEncoder().fit_transform(data['v22'])

encoded = pd.concat([data.drop(obj_columns, axis=1), onehots], axis = 1)

ss = StandardScaler()

# fit and transform data with PCA
pca = PCA(n_components = 325)
encoded_pca = pca.fit_transform(ss.fit_transform(encoded))
np.sum(pca.explained_variance_ratio_)

print('\n\n\n\n')

X_train, X_test, y_train, y_test = train_test_split(encoded_pca,
                                                    target,
                                                    test_size=0.33,
                                                    random_state=random_state)


# random search without CV
xgb_clf = xgb.XGBClassifier(nthread=32, random_state=random_state)
xgb_params = {
    'eta': np.linspace(0.01, 1, num=100, dtype='float'),
    'gamma':np.geomspace(0.001, 10, num=1000, dtype='float'),
    'max_depth':[d for d in range(1, 11)],
    'subsample':np.linspace(0.1, 1, num=100, dtype='float'),
    'colsample_bytree':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bynode':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'lambda': np.geomspace(0.001, 10, num=100, dtype='float'),
    'alpha': np.geomspace(0.001, 10, num=100, dtype='float')
}

search_iters = 1000

# sample param sets
param_sets = list()
for _ in range(search_iters):
    eta = random_state.choice(xgb_params['eta']),
    gamma = random_state.choice(xgb_params['gamma']),
    max_depth = random_state.choice(xgb_params['max_depth']),
    subsample = random_state.choice(xgb_params['subsample']),
    colsample_bytree = random_state.choice(xgb_params['colsample_bytree']),
    colsample_bylevel = random_state.choice(xgb_params['colsample_bylevel']),
    colsample_bynode = random_state.choice(xgb_params['colsample_bynode']),
    lambda_ = random_state.choice(xgb_params['lambda']),
    alpha = random_state.choice(xgb_params['alpha'])
    
    param_sets.append(
        {
            'eta': eta[0],
            'gamma': gamma[0],
            'max_depth': max_depth[0],
            'subsample': subsample[0],
            'colsample_bytree': colsample_bytree[0],
            'colsample_bylevel': colsample_bylevel[0],
            'colsample_bynode': colsample_bynode[0],
            'lambda': lambda_[0],
            'alpha': alpha
        }
    )

scores_log_loss = list()
scores_accuracy = list()

xgb_RSCV_start_time = time.time()

# loop over the random search parameters
for i, params in enumerate(param_sets):
    if i % 10 == 0:
        loop_time_s = time.time()
    xgb_clf.set_params(**params)
    cv = cross_validate(xgb_clf, X_train, y_train, 
               scoring={
                   'log_loss':log_loss_scorer,
                   'accuracy':accuracy_scorer
               })
    scores_log_loss.append(
        np.mean(cv['test_log_loss'])
    )
    scores_accuracy.append(
        np.mean(cv['test_accuracy'])
    )
    
    # print on each iteration just to see that something is running
    if i % 10 == 9:
        loop_time_e = time.time()
        duration = loop_time_e-loop_time_s
        print(f'{i+1}/{search_iters} Loop time:  {int(duration // 3600):02d}::{int((duration % 3600) // 60):02d}::{int((duration % 3600) % 60):02d}')
    sys.stdout.flush()

xgb_RSCV_end_time = time.time()
duration = xgb_RSCV_end_time-xgb_RSCV_start_time
# print the total run time
print(f'\nRandomized search done. {search_iters} iterations took \
{int(duration // 3600):02d}::{int((duration % 3600)//60):02d}::{int((duration % 3600) % 60):02d}')


# print the best parameters chosen by tuning
idx_best = np.argmin(scores_log_loss)
xgb_best_params_ = param_sets[idx_best]
print('\nBest Parameters')
print(xgb_best_params_)


print('\nSVM 5-fold Validation Performance')
# note test_log_loss is negated due to how scorers work 
# in parameter searches in sklearn
print('Mean Log Loss\t{}'.format(-scores_log_loss[idx_best]))
print('Mean Accuracy\t{}'.format(scores_accuracy[idx_best]))


# get performance on test set
xgb_clf.set_params(**param_sets[idx_best])
xgb_clf.fit(X_train, y_train)
xgb_y_test_pred = xgb_clf.predict(X_test)

print('\nSVM Test Set Performance')
print('Test Log Loss\t{}'.format(log_loss(xgb_y_test_pred, y_test)))
print('Test Accuracy\t{}'.format(accuracy_score(xgb_y_test_pred, y_test)))

print('\n\n\n')


# do some fit times for comparison with the SVM
cpuCount = os.cpu_count()
print(f'\nThere are {cpuCount} cores available.\n')
cores, sizes, times = list(), list(), list()
for n_cores in [1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 24, 32]:
    for size in [1000, 2000, 5000, 7000, 10000, 15000, 20000, 25000, 30000, 45000, 60000, 70000]:
        xgb_clf = xgb.XGBClassifier(random_state=random_state)
        xgb_clf.set_params(**xgb_best_params_)
        xgb_clf.set_params(**{'nthread': n_cores})
        sample = random_state.choice(np.arange(len(X_train)), size=size, replace=False)
        X_train_sub = X_train[sample, :]
        y_train_sub = y_train.iloc[sample]
        start_time = time.time()
        xgb_clf.fit(X_train_sub, y_train_sub)
        end_time = time.time()
        duration = end_time - start_time
        print(f'RF fit on {size} records with {n_cores} took {duration}')
        cores.append(n_cores)
        sizes.append(size)
        times.append(duration)

time_data = pd.DataFrame({
    'cores':cores,
    'sizes':sizes,
    'times':times})

# save the profile for later
time_data.to_csv('./data/XGB_profiling.csv')





