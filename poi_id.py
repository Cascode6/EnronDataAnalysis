#!/usr/bin/python

import os

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

import matplotlib.pyplot as plt

from time import time
import pprint
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary', 
                 'deferral_payments', 
                 'total_payments', 
                 'loan_advances', 
                 'bonus', 
                 'restricted_stock_deferred', 
                 'deferred_income', 
                 'total_stock_value', 
                 'expenses', 
                 'exercised_stock_options', 
                 'other', 
                 'long_term_incentive', 
                 'restricted_stock', 
                 'director_fees',
                 'to_messages',
                 'from_poi_to_this_person', 
                 'from_messages', 
                 'from_this_person_to_poi', 
                 'shared_receipt_with_poi'
                 ]



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
#Remove non-people entries in the dataset
data_dict.pop("TOTAL", 0)
data_dict.pop("THE_TRAVEL_AGENCY_IN_THE_PARK", 0)


### Task 3: Create new feature(s)
def makeFraction(poi, total):
    if poi != "NaN" and total != "NaN":
        converted_poi = float(poi)
        converted_total = float(total)
        ratio = converted_poi / converted_total
    else:
        ratio = "NaN"
    return ratio

for person in data_dict:
    from_poi = data_dict[person]["from_poi_to_this_person"]
    to_poi = data_dict[person]["from_this_person_to_poi"]
    from_total = data_dict[person]["from_messages"]
    to_total = data_dict[person]["to_messages"]

    data_dict[person]["from_emails_poi_ratio"] = makeFraction(from_poi,
                                                              from_total)
    data_dict[person]["to_emails_poi_ratio"] = makeFraction(to_poi, 
                                                            to_total)
    
features_list.append("from_emails_poi_ratio")
features_list.append("to_emails_poi_ratio")


### Store to my_dataset for easy export below.
removed_features = []
nan_percents = {} #to compare against feature predictive power later
for feature in features_list:
    total_nan = 0 
    total = 0
    for person in data_dict:
        total = total + 1
        if (feature, "NaN") in data_dict[person].items():
            total_nan = total_nan + 1
    nan_percents[feature] = float(total_nan) / float(total)
    if nan_percents[feature] > 0.4:
        removed_features.append(feature)
        features_list.remove(feature)
print removed_features
print features_list

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


#### for visualizing the actual and predicted outcomes
def visualizeData(features_test, labels_test, prediction):
    counter = 0
    for point in features_test: 
        salary = point[0]
        bonus = point[1]
        if labels_test[counter] == 0.:
            clr = "blue";
        elif labels_test[counter] == 1.:
            clr = "red"
        if salary != "NaN" and bonus != "NaN":
            plt.scatter( salary, bonus, color = clr)
        counter += 1

    plt.xlabel("salary")
    plt.ylabel("bonus")
    plt.show()

    try: 
        pred = prediction
        counter = 0
        for point in features_test:
            salary = point[0]
            bonus = point[1]
            if pred[counter] == 0.:
                clr = "blue";
            elif pred[counter] == 1.:
                clr = "red"
            if salary != "NaN" and bonus != "NaN":
                plt.scatter( salary, bonus, color = clr)
            counter += 1

        plt.xlabel("salary")
        plt.ylabel("bonus")
        plt.show()
    except NameError:
        pass
    return None


#allows for a comparison between feature Nan ratio and classifier importance
def get_Importances(classifier, filter):
    print "************************************"
    counter = 0
    for i in filter.get_support():
        if i == True:
            print features_list[counter]
        counter += 1
    print "************************************"
    return None

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(algorithm = "SAMME.R", 
                         learning_rate = 1.99)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# # Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.1, random_state=42)

# #taken from tester.py
from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(labels, 
                            n_iter=100, 
                            test_size=0.05,
                            random_state = 42)
for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
print "Total observations:", len(labels)
print "Total target observations:", sum(labels)
print "Total features:", len(features)
print "Number of train features:", len(features_train)
print "Number of train POIs:", sum(labels_train)
print "Number of test features:", len(features_test)
print "Number of test POIs:", sum(labels_test)


from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

sfm = SelectFromModel(clf)
sfm.fit_transform(features_train, labels_train)

#AdaBoostClassifier and SelectFromModel
PipeSelect = Pipeline([("select_from_model", sfm), ("adaboost", clf)])
t0 = time()
PipeSelect.fit(features_train, labels_train)
print "SFM training time:", round(time() - t0, 3), "s"
prediction = PipeSelect.predict(features_test)
print "SFM Score:", PipeSelect.score(features_test, labels_test)
#check the results
visualizeData(features_test, labels_test, prediction)
get_Importances(clf, sfm)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)