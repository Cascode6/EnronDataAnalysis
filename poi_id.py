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
    print feature
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
pprint.pprint(nan_percents)

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
print "n features", len(features_test[0])

#AdaBoost vs. KNeighbors vs. RandomForest
#and PCA vs. SelectPercentile
t0 = time()
print "****************************************"
# Adaboost and PCA pipe
BoosterPCA = Pipeline([("scaler", RobustScaler()),
                     ("pca", PCA()), ("booster", AdaBoostClassifier())])
BoosterPCA_params = {"pca__n_components": [3, 5, 7],
                 "booster__learning_rate":[.5, 1, 1.5],
                 "booster__n_estimators":[20, 50, 100]}

# AdaBoost and Select
BoosterPer = Pipeline([("select", SelectKBest()), ("booster", AdaBoostClassifier())])
BoosterPer_params = {"select__k": [3, 5, 7],
                    "booster__learning_rate":[.5, 1, 1.5],
                    "booster__n_estimators":[20, 50, 100]}

# KNearestNeighbors and PCA
KNearPCA = Pipeline([("scaler", RobustScaler()),
                     ("pca", PCA()), 
                     ("knear", KNeighborsClassifier())])
KNearPCA_params = {"pca__n_components": [3, 5, 7],
                   "knear__weights": ["uniform", "distance"],
                   "knear__n_neighbors":[3, 5, 7]}

# KNearestNeighbors and Select
KNearPer = Pipeline([("select", SelectKBest()), 
                     ("scaler", RobustScaler()),
                     ("knear", KNeighborsClassifier())])
KNearPer_params = {"select__k": [3, 5, 7],
                   "knear__weights": ["uniform", "distance"],
                   "knear__n_neighbors":[3, 5, 7]}

# RandomForest and PCA 
RForPCA = Pipeline([("scaler", RobustScaler()),
                     ("pca", PCA()), ("rforest", RandomForestClassifier())])
RForPCA_params = {"pca__n_components": [3, 5, 7],
                  "rforest__n_estimators":[5, 10, 20],
                  "rforest__class_weight":[None, "balanced"]}

# RandomForest and SelectPercentile
RForPer = Pipeline([("select", SelectKBest()), ("rforest", RandomForestClassifier())])
RForPer_params = {"select__k": [3, 5, 7],
                  "rforest__n_estimators":[5, 10, 20],
                  "rforest__class_weight":[None, "balanced"]}

# create classifier/parameter dict list of lists 
testPipes = [[BoosterPCA, BoosterPCA_params], 
           [BoosterPer, BoosterPer_params], 
           [KNearPCA, KNearPCA_params],
           [KNearPer, KNearPer_params],
           [RForPCA, RForPCA_params],
           [RForPer, RForPer_params]]


testCls = [AdaBoostClassifier(),
           (Pipeline([("scaler", RobustScaler()),
                     ("knei", KNeighborsClassifier())])),
           RandomForestClassifier()]

for test in testCls:
    cl = test.fit(features_train, labels_train)
    print "------Base Classifier Score------"
    print test
    print classification_report(labels_test, cl.predict(features_test))
    accuracy, precision, recall, f1 = test_classifier(cl, my_dataset, features_list)

scoring = ["accuracy", "precision", "recall", "f1"]

def getBestClassifier(testPipes, scoring):
    top_scorers = []
    for scorer in scoring:
        best_classifiers = []
        best_scores = []
        # initialize grid search with pipeline (cl[0]) and parameters (cl[1])
        for cl in testPipes:
            grid = GridSearchCV(cl[0], param_grid = cl[1], cv = cv, scoring = scorer)
            #based on https://discussions.udacity.com/t/gridsearchcv-and-stratifiedshufflesplit-giving-indexerror-list-index-out-of-range/39018/8?u=cfactoidal
            estimator = grid.fit(features, labels)
            print "-------Tuned Estimator-------"
            print estimator.best_estimator_
            print scorer, "score:", estimator.best_score_
            best_classifiers.append(estimator.best_estimator_)
            best_scores.append(estimator.best_score_)

        best_score = max(best_scores)
        best_score_index = best_scores.index(best_score)
        print "--------------------------------"
        print "Best", scorer, "scoring model:"
        bestPipe = best_classifiers[best_score_index].fit(features_train, labels_train)
        top_scorers.append(bestPipe)
        print "GridSearchCV", scorer, "score:", best_score
    return top_scorers

top_scorers = getBestClassifier(testPipes, scoring)
scores_by_model = []
best_recall = 0
counter = 0
for est in top_scorers:
    print "~~~~~~for", scoring[counter], ": ~~~~~~~~" 
    real_accuracy, real_precision, real_recall, real_f1 = test_classifier(est, my_dataset, features_list)
    if real_recall > best_recall:
        best_recall = real_recall
        clf = est
    scores_by_model.append([est, real_accuracy, real_precision, real_recall, real_f1])
    counter +=1

#parse feature weights in case highest-performing recall score is PCA
def getWeights(components, features_list):
        counter = 0
        while counter < 15:
            print features_list[counter + 1], round(components[counter], 5)
            counter = counter + 1
        return None
try:
    imps = clf.named_steps["select"].scores_
    counter = 1
    for imp in imps:
        print features_list[counter], imp
        counter +=1
except:
    weights1 = Tested.named_steps["pca"].components_[0]
    weights2 = Tested.named_steps["pca"].components_[1]
    weights3 = Tested.named_steps["pca"].components_[2]
    print "--------------------------------"
    print "Feature weights in component 1:"
    getWeights(weights1, features_list)
    print "Feature weights in component 2:"
    getWeights(weights2, features_list)
    print "Feature weights in component 3:"
    getWeights(weights3, features_list)

print "****************************************"
print "run time:", round(time() - t0, 3), "s"


# ###Task 6: Dump your classifier, dataset, and features_list so anyone can
# ###check your results. You do not need to change anything below, but make sure
# ###that the version of poi_id.py that you submit can be run on its own and
# ###generates the necessary .pkl files for validating your results.

print "Best balanced estimator so far:"
test_classifier(clf, my_dataset, features_list)

dump_classifier_and_data(clf, my_dataset, features_list)