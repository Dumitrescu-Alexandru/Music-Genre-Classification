# Music-Genre-Classification

Predict a music genre given ”rhythm”, ”MFCC” and ”chroma” patterns and a label as the music genre to which each data point belongs. Some of the models implemented were fully connected NNs with gridsearch for hyperparameters (TensorFlow), SVC, Random Forest, XGBoost (Scikit-learn).

## Summary

To tackle the problem of classifying the musical genre of songs based on "carefully chosen features", approaches using support vector machines (SVM), Random Forest, XGBoost, and deep neural networks were used.

Data analysis showed high class imbalance in the dataset, and visualizations such as the correlation matrix of features, 3-dimensional PCA projection, and feature-class plots, indicate that many of the features are possibly non-informative and that classes are not easily separable.

On the Kaggle dataset, the best variants of all models ranged from 60.2%-66.2% in accuracy (33rd out of 406) and 0.236-0.171 in log-loss (38th out of 371), with XGBoost scoring the highest on both metrics.

