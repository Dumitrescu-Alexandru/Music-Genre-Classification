import pandas as pd
import MLBP.batches
import numpy as np
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder


class SVM_classifiers():
    data = MLBP.batches.Batches(feature_scaling=True)
    def __init__(self):
        clf = svm.SVC(gamma='scale',decision_function_shape='ovo')
        self.data.smote()
        # clf.fit(self.data.data_train,np.reshape(self.data.train_labels,self.data.train_labels.shape[0]))
        # predictions = clf.predict(self.data.data_test)
        # sum = np.zeros(11)
        # for i in predictions:
        #     sum[i]+=1
        # print(sum)
        #
        # clf2 = svm.LinearSVC(max_iter=1000000000)
        # clf2.fit(self.data.data_train, np.reshape(self.data.train_labels, self.data.train_labels.shape[0]))
        # predictions = clf2.predict(self.data.data_test)
        # sum = np.zeros(11)
        # for i in predictions:
        #     sum[i] += 1
        # print(sum)

        # clf3 = svm.SVC(kernel='linear')
        # clf3.fit(self.data.data_train, np.reshape(self.data.train_labels, self.data.train_labels.shape[0]))
        # predictions = clf3.predict(self.data.data_test)
        # sum = np.zeros(11)
        # for i in predictions:
        #     sum[i] += 1
        # print(sum)

        clf4 = svm.SVC(kernel='poly',degree=4)
        clf4.fit(self.data.data_train, np.reshape(self.data.train_labels, self.data.train_labels.shape[0]))
        predictions = clf4.predict(self.data.data_test)
        sum = np.zeros(11)
        for i in predictions:
            sum[i] += 1
        print(sum)
        tpot_pred = pd.read_csv(r"C:\Users\alex\Desktop\aalto courses\MLBP\project\pycharm_proj\MLBP\results_kaggle\tpot_accuracy_solution.csv")
        tpot_vals = tpot_pred.values
        tpot_vals = tpot_vals[:,1]
        matched = 0
        for i in range(tpot_vals.shape[0]):
            if tpot_vals[i] == predictions[i]:
                matched+=1
        print(matched)
a = SVM_classifiers()