import time
import pprint
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
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

import time
import pprint
import os

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

log_loss_scorer = make_scorer(log_loss, greater_is_better=False)
accuracy_scorer = make_scorer(accuracy_score)

random_state = np.random.RandomState(42)

# Get the number of CPUs 
# in the system using 
# os.cpu_count() method 
cpuCount = os.cpu_count() 
  
# Print the number of 
# CPUs in the system 
print("Number of CPUs in the system:", cpuCount)
sys.stdout.flush()



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

X_train, X_test, y_train, y_test = train_test_split(encoded_pca,
                                                    target,
                                                    test_size=0.33,
                                                    random_state=random_state)


print(f'The shape of X_train is {X_train.shape[0]}x{X_train.shape[1]}')



rf_clf = RandomForestClassifier(random_state=random_state)
rf_params = {
    'n_estimators': np.linspace(10, 150, dtype='int'),
    'criterion':['gini', 'entropy'],
    'max_depth': np.linspace(10, 100, dtype='int'),
    'min_samples_split': np.linspace(2, 100, 50, dtype='int'),
    'min_samples_leaf': np.linspace(2, 100, 50, dtype='int'),
    'max_features': ['auto', 'sqrt', 'log2']
}

search_iters = 1000

rf_RSCV_start_time = time.time()
# setup search
rf_RSCV = RandomizedSearchCV(rf_clf, rf_params, scoring=log_loss_scorer,
                             n_iter=search_iters, random_state=random_state,
                             n_jobs=32)
# seach
rf_RSCV.fit(X_train, y_train)

rf_RSCV_end_time = time.time()
duration = rf_RSCV_end_time-rf_RSCV_start_time

print(f'Randomized CV search done. {search_iters} iterations took \
{int(duration // 3600):02d}::{int((duration % 3600) // 60):02d}::{int((duration % 3600) % 60):02d}')

# print the best parameters chosen by CV
pprint.pprint(rf_RSCV.best_params_)

# get CV results with best parameters
rf_clf.set_params(**rf_RSCV.best_params_)
rf_cv = cross_validate(rf_clf, X_train, y_train, n_jobs=32,
                       scoring={
                           'log_loss':log_loss_scorer,
                           'accuracy':accuracy_scorer
                       })


print('RF 5-fold Validation Performance')
# note test_log_loss is negated due to how scorers work 
# in parameter searches in sklearn
print('Mean Log Loss\t{}'.format(np.mean(-rf_cv['test_log_loss'])))
print('Mean Accuracy\t{}'.format(np.mean(rf_cv['test_accuracy'])))


# get performance on test set
rf_clf.fit(X_train, y_train)
rf_y_test_pred = rf_clf.predict(X_test)

print('RF Test Set Performance')
print('Test Log Loss\t{}'.format(log_loss(rf_y_test_pred, y_test)))
print('Test Accuracy\t{}'.format(accuracy_score(rf_y_test_pred, y_test)))


# do some fit times for comparison with the SVM
cpuCount = os.cpu_count()
print(f'There are {cpuCount} cores available.\n')
cores, sizes, times = list(), list(), list()
for n_cores in [1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 24, 32]:
    for size in [1000, 2000, 5000, 7000, 10000, 15000, 20000, 25000, 30000, 45000, 60000, 70000]:
        rf_clf = RandomForestClassifier(random_state=random_state)
        rf_clf.set_params(**rf_RSCV.best_params_)
        rf_clf.set_params(**{'n_jobs': n_cores})
        sample = random_state.choice(np.arange(len(X_train)), size=size, replace=False)
        X_train_sub = X_train[sample, :]
        y_train_sub = y_train.iloc[sample]
        start_time = time.time()
        rf_clf.fit(X_train_sub, y_train_sub)
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
time_data.to_csv('./data/RF_profiling.csv')


