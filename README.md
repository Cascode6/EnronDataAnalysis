# EnronDataAnalysis

<h1>Enron Data Analysis with AdaBoost and SelectFromModel</h1>

The Enron email corpus is one of the most famous in data science and machine learning. My investigation of the features provided by Udacity's ML Final Project was an attempt to sift out meaningful trends and, notably, to achieve precision and recall scores above 0.3 -- this report will detail my process, results and commandline output towards those goals.

<h1>Conclusion</h1>
Utilized a PCA/Adaboost pipeline with default parameters and n_components = 3, StratifiedShuffleSplit cross validation with test_size=0.2, and achieved the following scores:

Accuracy: 0.85793	Precision: 0.45517	Recall: 0.3325F1: 0.38428	F2: 0.35144
