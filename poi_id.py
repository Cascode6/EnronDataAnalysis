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
                 'bonus',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'total_stock_value',
                 'shared_receipt_with_poi',
                 'exercised_stock_options',
                 'to_messages',
                 'from_messages',
                 'loan_advances',
                 'director_fees',
                 'expenses',
                 'other',
                 'total_payments',
                 'deferral_payments',
                 'long_term_incentive',
                 'from_this_person_to_poi',
                 'from_poi_to_this_person',
                 'deferred_income',
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


######### eventually, create text-based count features #################

### Store to my_dataset for easy export below.

# n_people = 0
# for person in data_dict:
#     n_people = n_people + 1
# print "Number of people in dataset: ", n_people
# print "Number of features used: ", len(features_list)
# print "Total number of datapoints: ", n_people * len(features_list)
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
    if nan_percents[feature] >= 0.75:
        removed_features.append(feature)
        features_list.remove(feature)
# print removed_features
# print features_list

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


####visualization function
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






#allows for a comparison between percent of NaN values and the importance to the classifier
def get_Importances(classifier, filter):
    print "************************************"
    
    # imps = classifier.feature_importances_
    counter = 0
    # truecount = 0
    # for i in filter.get_support():
    #     feature_name = features_list[counter]
    #     if i == True:
    #         print feature_name, "NaN:", nan_percents[feature_name], "Imp:", imps[truecount]
    #         truecount +=1
    #     counter += 1
    for i in filter.get_support():
        if i == True:
            print features_list[counter]
        counter += 1
    print "************************************"
    return None

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans

# dt = DecisionTreeClassifier()


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
cv = StratifiedShuffleSplit(labels, n_iter=100, test_size=0.05, random_state = 42)
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
# print "Number of train features:", len(features_train)
# print "Number of train POIs:", sum(labels_train)
# print "Number of test features:", len(features_test)
# print "Number of test POIs:", sum(labels_test)



# from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


# ###Decision Tree Classifier Only
# sfm = SelectFromModel(dt)
# sfm.fit(features_train, labels_train)
# Feature_transform = sfm.transform(features_train)
# PipeSelect = Pipeline([("select_from_model", sfm), ("dtc", dt)])
# t0 = time()
# PipeSelect.fit(features_train, labels_train)
# print "Decision Tree training time:", round(time() - t0, 3), "s"
# prediction = PipeSelect.predict(features_test)
# print "Decision Tree Score:", PipeSelect.score(features_test, labels_test)
# #check the results
# visualizeData(features_test, labels_test, prediction)
# get_Importances(dt, sfm)


# sfm = SelectFromModel(clf)
# sfm.fit(features_train, labels_train)



#from PCA lesson:
def doPCA(n_components):
    pca = PCA(n_components = n_components)
    pca.fit(data)
    return pca

n_components = 2
pca = doPCA(n_components)
print pca.explained_variance_ratio_
first_pca = pca.components_[0]
second_pca = pca.components_[1]
# third_pca = pca.components_[2]

transformed_data = pca.transform(data)




# #AdaBoostClassifier and SelectFromModel
# PipeSelect = Pipeline([("select_from_model", sfm), ("adaboost", clf)])
# t0 = time()
# PipeSelect.fit(features_train, labels_train)
# print "SFM training time:", round(time() - t0, 3), "s"
# prediction = PipeSelect.predict(features_test)
# print "SFM Score:", PipeSelect.score(features_test, labels_test)
# #check the results
# visualizeData(features_test, labels_test, prediction)
# get_Importances(clf, sfm)

print "n pca features:", n_components
PipeSelect = Pipeline([("pca", pca), ("adaboost", clf)])
t0 = time()
PipeSelect.fit(features_train, labels_train)
print "pca training time:", round(time() - t0, 3), "s"
prediction = PipeSelect.predict(features_test)
print "pca Score:", PipeSelect.score(features_test, labels_test)

#check the results
for ii, jj in zip(transformed_data, data):
    plt.scatter(first_pca[0]*ii[0], first_pca[1]*ii[0], color="r")
    plt.scatter(second_pca[0]*ii[0], second_pca[1]*ii[0], color="c")
    # plt.scatter(third_pca[0]*ii[0], third_pca[1]*ii[0])
    plt.scatter(jj[0], jj[1], color="b")
    
plt.show()

visualizeData(features_test, labels_test, prediction)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)