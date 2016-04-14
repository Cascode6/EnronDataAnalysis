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
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


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
    sent_from_poi = data_dict[person]["from_poi_to_this_person"]
    sent_to_poi = data_dict[person]["from_this_person_to_poi"]
    sent_total = data_dict[person]["from_messages"]
    recieved_total = data_dict[person]["to_messages"]

    data_dict[person]["from_emails_poi_ratio"] = makeFraction(sent_from_poi,
                                                              recieved_total)
    data_dict[person]["to_emails_poi_ratio"] = makeFraction(sent_to_poi, 
                                                            sent_total)
    
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
###gets passed the testing data and labels, the clf prediction, and
#  a list of lists containing the used features, their index point &
#  their informational importance from get_Importances()
#  can be used with changing no.s of features
def visualizeData(features_test, labels_test, features_used, prediction):
    counter = 0
    print "No. of features used:",len(features_used)
    point1 = features_used[0]
    point2 = features_used[1]
    for feat in features_used: #compare the importances
        print feat[2]
        if feat[2] > point2[2]:
            if feat[2] > point1[2]:
                point1 = feat
            else:
                point2 = feat
        else:
            pass
    print point1[0], point1[2], point2[0], point2[2]
    p1index = point1[1] #set the index point for use with features_test
    p2index = point2[1]
    
    for data_point in features_test: #iterates through each observations' feature array
        if labels_test[counter] == 0.:
            clr = "blue";
        elif labels_test[counter] == 1.:
            clr = "red" #sets color by actual POI id
        plt.scatter( data_point[p1index], data_point[p2index], color = clr)
        counter += 1

    plt.xlabel(point1[0])
    plt.ylabel(point2[0])
    plt.show()

    try: 
        counter = 0
        for data_point in features_test:
            if prediction[counter] == 0.:
                clr = "blue";
            elif prediction[counter] == 1.:
                clr = "red" #sets color by predicted POI id
            plt.scatter( data_point[p1index], data_point[p2index], color = clr)
            counter += 1
            
        plt.xlabel(point1[0])
        plt.ylabel(point2[0])
        plt.show()
    except NameError:
        pass
    return None


#allows for a comparison between feature Nan ratio and classifier importance
def get_Importances(classifier, filter):
    print "************************************" #for console log
    counter = 0
    true_counter = -1
    features_used = []
    importances = classifier.feature_importances_ #importance of features used
    for i in filter.get_support(): #whether feature was used or not
        if i == True:
            true_counter += 1
            print features_list[counter], importances[true_counter]
            features_used.append([features_list[counter], 
                                  counter,
                                  classifier.feature_importances_[true_counter]])
        else:
            pass
        counter += 1
    print "************************************"
    return features_used

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(learning_rate = 0.3, n_estimators = 100)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# # # Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.1, random_state=42)

#taken from tester.py, adusted for better results
from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(labels, 
                            n_iter=100000, #tuned
                            test_size=0.05,#tuned
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

from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA


#AdaBoostClassifier and SelectFromModel Pipeline
sfm = SelectFromModel(clf)
sfm.fit_transform(features_train, labels_train)
PipeSelect = Pipeline([("select_from_model", sfm), ("adaboost", clf)])
t0 = time()
PipeSelect.fit(features_train, labels_train)
print "SFM training time:", round(time() - t0, 3), "s"
prediction = PipeSelect.predict(features_test)
print "SFM Score:", PipeSelect.score(features_test, labels_test)

#check the results
features_used = get_Importances(clf, sfm)
visualizeData(features_test, labels_test, features_used, prediction)

################# feature/classifier selection process used ############
# from sklearn.feature_selection import SelectPercentile, SelectKBest
# ### Report Investigations with Base
# base_DT.fit(features_train, labels_train)
# print "Base DT accuracy:", base_DT.score(features_test, labels_test)
# base_ada.fit(features_train, labels_train)
# print "Base ADA accuracy:", base_ada.score(features_test, labels_test)

# FilterModelDT = SelectFromModel(base_DT)
# FilterModelAda = SelectFromModel(base_ada)
# t0 = time()
# FilterModelDT.fit_transform(features_train,labels_train)
# print "DTModel fit time:", round(time() - t0, 3), "s"
# t0 = time()
# FilterModelAda.fit_transform(features_train,labels_train)
# print "AdaModel fit time:", round(time() - t0, 3), "s"

# percent = 10
# FilterPercentile = SelectPercentile(percentile=percent)
# t0 = time()
# FilterPercentile.fit_transform(features_train,labels_train)
# print "Percentile training time:", round(time() - t0, 3), "s"

# K = 2
# FilterKBest = SelectKBest(k=K)
# t0=time()
# FilterKBest.fit_transform(features_train,labels_train)
# print "KBest training time:", round(time() - t0, 3), "s"

# def do_PCA(n_comps):
#     pca = PCA(n_components = n_comps)
#     pca.fit_transform(features_train, labels_train)
#     return pca
    
# n_comps = 2
# t0 = time()
# pca = do_PCA(n_comps)
# print "PCA training time:", round(time() - t0, 3), "s"

# filters = [FilterPercentile, 
#            FilterKBest,
#            pca]

# dt_pipeline = make_pipeline(FilterModelDT, base_DT)
# dt_pipeline.fit(features_train, labels_train)
# print FilterModelDT, "Accuracy DT:", dt_pipeline.score(features_test, labels_test)
# print "*****************************"
# dt_pipeline = make_pipeline(FilterModelAda, base_ada)
# dt_pipeline.fit(features_train, labels_train)
# print FilterModelAda, "Accuracy ADA:", dt_pipeline.score(features_test, labels_test)
# print "*****************************"

# for i in filters:
#     dt_pipeline = make_pipeline(i, base_DT)
#     dt_pipeline.fit(features_train, labels_train)
#     print i, "Accuracy DT:", dt_pipeline.score(features_test, labels_test)
#     print "*****************************"
#     ada_pipeline = make_pipeline(i, base_ada)
#     ada_pipeline.fit(features_train, labels_train)
#     print i, "Accuracy ADA:", ada_pipeline.score(features_test, labels_test)
#     print "*****************************"
    
    
# clf = make_pipeline(pca, base_DT)
# clf.fit(features_train, labels_train)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)