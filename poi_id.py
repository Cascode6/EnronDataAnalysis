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

# email to/from poi ratio
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
    
    #financial stat based on stock
    ts = data_dict[person]["total_stock_value"]
    eso = data_dict[person]["exercised_stock_options"]
    
    
    data_dict[person]["from_emails_poi_ratio"] = makeFraction(sent_from_poi,
                                                              recieved_total)
    data_dict[person]["to_emails_poi_ratio"] = makeFraction(sent_to_poi, 
                                                            sent_total)
    
    if ts not in ["NaN", 0] and eso not in ["NaN", 0]:
        data_dict[person]["multiplied_stock"] = ts * eso
        data_dict[person]["exercised_stock_sq"] = eso^2
    else:
        data_dict[person]["multiplied_stock"] = "NaN"
        data_dict[person]["exercised_stock_sq"] = "NaN"

features_list.append("multiplied_stock")
features_list.append("exercised_stock_sq")
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

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#taken from tester.py and tuned
from sklearn.cross_validation import StratifiedShuffleSplit
cv = StratifiedShuffleSplit(labels,         
                            test_size=0.2,         
                            random_state = 66)
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

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report


#AdaBoost and PCA pipeline
AdaPCA = Pipeline([("pca", PCA(n_components=3)),
                 ("ada", AdaBoostClassifier())])
clf = AdaPCA.fit(features_train, labels_train)
print 'Ada/PCA'
print classification_report(labels_test, clf.predict(features_test))
print ''
test_classifier(clf, my_dataset, features_list)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)