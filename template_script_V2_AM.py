"""
    Python script to submit as a part of the project of ELTP 2020 course.
    
    This script serves as a template. Please use proper comments and meaningful variable names.
"""

"""
    Group Members:
        (1) Camille Morand Duval
        (2) Niels Nicolas
        (3) Ariel Modai
"""

"""
    i) Import necessary packages
"""

import numpy as np
import pandas as pd
import spacy
import fr_core_news_sm

# Choose number of samples tested
import math

# import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# dimension reduction
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import SparsePCA

# plots
import matplotlib.pyplot as plt
import seaborn as sns

# pick at random 10% of the total samples
import random

# classifiers
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

# hyperparameters tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import RandomizedSearchCV

"""
    ii) Load spaCy for french.
"""

spacy_nlp = fr_core_news_sm.load()

"""
    iii) Download data.
"""

X_train = pd.read_csv('X_train.csv')
Y_train = pd.read_csv('Y_train.csv')
X_test = pd.read_csv('X_test.csv')

"""
    Pre-processing
"""


"""
    Your methods implementing the models.
    
    Each of your model should have a separate method. e.g. run_random_forest, run_decision_tree etc.
    
    Your method should:
        (1) create the proper instance of the model with the best hyperparameters you found
        (2) fit the model with a given training data
        (3) run the prediction on a given test data
        (4) return accuracy and F1 score
        
    Following is a sample method. Please note that the parameters given here are just examples.
"""
def model_random_forest(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = RandomForestClassifier(n_estimators = 100, max_depth=2, random_state=0) # please choose all necessary parameters
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1


"""
   The main function should print all the accuracies and F1 scores for all the models.
   
   The names of the models should be sklearn classnames, e.g. DecisionTreeClassifier, RandomForestClassifier etc.
   
   Please make sure that your code is outputting the performances in proper format, because your script will be run automatically by a meta-script.
"""
if __name__ == "__main__":
    """
       This is just an example, plese change as necessary. Just maintain final output format with proper names of the models as described above.
    """
    model_1_acc, model_1_f1 = run_model_1(...)
    model_2_acc, model_2_f1 = run_model_2(...)
    """
        etc.
    """

    # print the results
    print("model_1", model_1_acc, model_1_f1)
    print("model_2", model_2_acc, model_2_f1)
    """
        etc.
    """
