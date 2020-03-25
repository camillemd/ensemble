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
import en_core_web_sm
import de_core_news_sm

# Choose number of samples tested
import math

# import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import re 
"""
    ii) Load spaCy for french.
"""

spacy_nlp = fr_core_news_sm.load()

"""
    iii) Download data.
"""

X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/Y_train.csv')
X_test = pd.read_csv('data/X_test.csv')

#for testing if the model is running - remove before hand-in
#remove 205 / 220 - uncomment 206 / 221
X_train = X_train[:1000]
y_train = y_train[:1000]
X_test = X_test[:100]
y_test = y_train[:100]

"""
    Pre-processing
"""
# designation and ids
def cleaning(X_train): 
    X_train = X_train.drop(['description', 'productid','imageid'], axis=1)
    X_train.columns = ['integer_id', 'designation']
    return X_train

def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')

    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')

    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')
    
    return string
 
def del_digits(list): 
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list] 
    return list

def del_date(list): 
    pattern = '^(?:(?:[0-9]{2}[:\/,]){2}[0-9]{2,4}|am|pm)$'
    list = [re.sub(pattern, '', i) for i in list] 
    return list

def del_specialchar(list): 
    pattern = '\W'
    list = [re.sub(pattern, '', i) for i in list] 
    return list

def del_letters(list): 
    for i in list:
      if len(i) <= 1:
        list.remove(i)
    return list

def raw_to_tokens(raw_string, spacy_nlp):
    # Write code for lower-casing
    string = raw_string.lower()
    
    # Write code to normalize the accents
    string = normalize_accent(string)
        
    # Write code to tokenize
    spacy_tokens = spacy_nlp(string)
        
    # Write code to remove punctuation tokens and create string tokens
    string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct if not token.is_stop]

    # Write code to remove digits
    string_tokens = del_digits(string_tokens)

    # Write code to remove special character
    string_tokens = del_specialchar(string_tokens)

    # Write code to remove dates
    string_tokens = del_date(string_tokens)

    # Write code to remove single letters
    string_tokens = del_letters(string_tokens)
    
    # Write code to join the tokens back into a single string
    clean_string = " ".join(string_tokens)

    #return clean_string
    return clean_string

## Apply pre-processing

# X_train - step takes roughly ~15:30 min 
X_train = cleaning(X_train)
spacy_nlp = fr_core_news_sm.load() 
X_train['designation_cleaned'] = X_train['designation'].apply(lambda x: raw_to_tokens(x, spacy_nlp))
spacy_nlp = spacy.load('en_core_web_sm')
X_train['designation_cleaned'] = X_train['designation_cleaned'].apply(lambda x: raw_to_tokens(x, spacy_nlp))
spacy_nlp = de_core_news_sm.load()
X_train['designation_cleaned'] = X_train['designation_cleaned'].apply(lambda x: raw_to_tokens(x, spacy_nlp))

# X_test - step takes roughly ~2:20 min 
X_test = cleaning(X_test)
spacy_nlp = fr_core_news_sm.load()
X_test['designation_cleaned'] = X_test['designation'].apply(lambda x: raw_to_tokens(x, spacy_nlp))
spacy_nlp = spacy.load('en_core_web_sm')
X_test['designation_cleaned'] = X_test['designation_cleaned'].apply(lambda x: raw_to_tokens(x, spacy_nlp))
spacy_nlp = de_core_news_sm.load()
X_test['designation_cleaned'] = X_test['designation_cleaned'].apply(lambda x: raw_to_tokens(x, spacy_nlp))

# create a list from the processed cells
doc_clean_train =  X_train['designation_cleaned'].astype('U').tolist()
doc_clean_test = X_test['designation_cleaned'].astype('U').tolist()
#doc_clean = doc_clean_train + doc_clean_test
doc_clean = doc_clean_train + doc_clean_test

# import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# X-train
# convert raw documents into TF-IDF matrix.
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(doc_clean)
print("Shape of the TF-IDF Train Matrix:")
print(X_tfidf.shape)

# if no transformation is applied i.e. no PCA / truncated SVD: 
X_transformed = X_tfidf
X_train_T = X_transformed[:1000]
X_test_T = X_transformed[1000:]
#X_train_T = X_transformed[:84916]
print(X_train_T.shape) # 84916
print(X_test_T.shape) # 13812

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
def model_decision_classifier(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = DecisionTreeClassifier(criterion="gini", 
                                 max_depth=180, 
                                 splitter = "random") # please choose all necessary parameters
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1

def model_random_forest(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = RandomForestClassifier(n_estimators = 1800, 
                                 min_samples_split = 5, 
                                 min_samples_leaf = 1, 
                                 max_features = "auto", 
                                 max_depth = None, 
                                 bootstrap = True) # please choose all necessary parameters
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1

def model_boosting(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = xgb.XGBClassifier(tree_method = "approx", 
                            subsample = 0.8, 
                            silent = False,
                            objective = "multi:softmax",
                            num_classes = 27, 
                            min_child_weight = 1, 
                            max_depth = 10, 
                            max_delta_step = 0,
                            eta = 0.15,
                            colsample_bytree = 0.8,
                            colsample_bylevel = 1,
                            booster = "gbtree") # please choose all necessary parameters
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1

def model_grad_boosted_trees(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = GradientBoostingClassifier(n_estimator = 300,
                                     max_depth = 8,
                                     learning_rate = 0.15) # please choose all necessary parameters
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1

def model_bagging(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    
    parameters = {"base_estimator__criterion" : ["gini", "entropy"],
                  "base_estimator__splitter" :   ["best", "random"],
                  "n_estimators": [10, 20, 30], 
                  "max_samples" : [0.7, 0.8, 1]
                  }

    DTC = DecisionTreeClassifier(random_state = 11, 
                                 max_features = 0.8,
                                 max_depth = 6)

    clf = BaggingClassifier(base_estimator = DTC) # please choose all necessary parameters
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1

def model_adaboost(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    
    parameters = {"n_estimators" : 100, 
                  "learning_rate" : 0.1}

    RFC = RandomForestClassifier(criterion = 'gini', 
                             n_estimators = 2000, 
                             min_samples_split = 10, 
                             min_samples_leaf = 2, 
                             max_features = 'sqrt', 
                             max_depth = None, 
                             bootstrap = True)

    clf = BaggingClassifier(base_estimator = RFC) # please choose all necessary parameters
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
       This is just an example, please change as necessary. Just maintain final output format with proper names of the models as described above.
    """
    model_1_acc, model_1_f1 = model_decision_classifier(X_train, y_train, X_test, y_test)
    model_2_acc, model_2_f1 = model_random_forest(X_train, y_train, X_test, y_test)
    model_3_acc, model_3_f1 = model_boosting(X_train, y_train, X_test, y_test)
    model_4_acc, model_4_f1 = model_grad_boosted_trees(X_train, y_train, X_test, y_test)
    model_5_acc, model_5_f1 = model_bagging(X_train, y_train, X_test, y_test)
    model_6_acc, model_6_f1 = model_adaboost(X_train, y_train, X_test, y_test)
        
        
    #model_1_acc, model_1_f1 = run_model_1(...)
    """
        etc.
    """

    # print the results
    print("DecisionTreeClassifier", model_1_acc, model_1_f1)
    print("RandomForestClassifier", model_2_acc, model_2_f1)
    print("XGBClassifier", model_2_acc, model_2_f1)
    print("GradientBoostingClassifier", model_2_acc, model_2_f1)
    print("BaggingClassifier", model_2_acc, model_2_f1)
    print("AdaBoostClassifier", model_2_acc, model_2_f1)
    """
        etc.
    """
