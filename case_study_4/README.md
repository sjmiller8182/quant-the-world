# Random Forest, XGBoost, and SVM

## Summary

Random Forest, eXtreme Gradient Boosting (XGBoost), and Support Vector Machine (SVM) are three common machine learning models used in predictive analytics. Random Forest is a bootstrap-aggregated tree-based model that builds ensemble models with weak-learning decision trees where each tree is created using the values of an independent set of random vectors derived from a fixed probability distribution. The Random Forest model is considered difficult to over-fit, easily interpretable, and very reliable.

eXtreme Gradient Boosting is a model that creates a partition tree to make predictions on class-level outcomes using data subsets. New, subsequent partition trees are applied to remaining batches of the data set until residual error is minimized. The weight of each sample batch is adaptively changed after each round of boosting (new tree) such that the model focuses on building trees to correctly explain data contributing to incorrect classifications. This is repeated until optimal performance is obtained. However, XGBoost is prone to over-fitting.

The third model, Support Vector Machine, operates by searching for a hyperplane - a linear decision boundary - that provides the largest margin of separation between classes in a data set. SVM is therefore referred to as a maximal margin classifier. SVM maximizes the margin by minimizing the objective loss function. Because the objective loss function is quadratic and the constraints are linear, SVM faces what is called a convex optimization problem that it solves using a Lagrange multiplier method.

This study applies Random Forest, XGBoost, and an SVM to the same data set and the models are then compared in terms of processing speed, model accuracy, and model loss. Speed is measured across a 2.1GHz processor with 32 cores. Because SVM is not parallelizable, core performance testing is fixed across only one processor core for the SVM.

[See Full Report](./Random Forest, XGBoost, and SVM.pdf)

**Language**: Python

## Data

Data was provided by the instructor. 
It is included in the data folder.


