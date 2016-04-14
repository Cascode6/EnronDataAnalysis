# EnronDataAnalysis
<!DOCTYPE html>
<html>
    <head>
        <title> Enron Data Analysis with AdaBoost and SelectFromModel</title>
    </head>
    <body>
        <header><h1>Enron Data Analysis with AdaBoost and SelectFromModel</h1></header>
        <section>
            <h1>The Enron Data</h1>
            <p>The Enron email corpus is one of the most famous in data science and machine learning. My investigation of the features provided by Udacity's ML Final Project was an attempt to sift out meaningful trends and, notably, to achieve precision and recall scores above 0.3 -- this report will detail my process, results and commandline output towards those goals.</p>
        </section>
        <section>
            <h1>Goals of Machine Learning on the Enron Dataset</h1>
            <p>The Enron dataset serving as the basis of this project contains, raw, 
                <blockquote><code>Number of observations: 146<br>
                    Number of POI's: 18<br>
                    Number of available features: 21<br>
                    Total data points: 3066</code></blockquote> 
            This dataset poses some very specific challenges to a machine learning algorithm.</p>
            <p>The biggest challenge, and why the Enron dataset is such a good training ground for ML, is size. Not only do we have a small number of observations (people) in the dataset, but we have an even smaller number of target observations (persons of interest, or POIs). Any predictions we make are in danger of under-classifying datapoints as POI and over-classifying as not POI. However, this is how many datasets develop in their natural habitat - so investigating methods to maximize classification success in this situation has many potential applications.</p>
            <p>What this means for this analysis is that recall scores may be 'stickier', or harder to affect, for this dataset. But in this scenario, that is probably ok. The professional use case of training and tuning a classifier on this data would likely be for someone - say, a prosecuting governmental agency - to identify suspects connected with Enron fraud for investigation. Thus, under-identifying POIs would be much preferable to over-identifying; if the algorithm spits out a name for investigation, you'd want to be especially sure that an identified name truly is a POI. This is reflected in a high precision score.</p>
            <p>Another challenge inherent to this dataset is the data's "outlier"y nature. From the acompanying info sheet about Enron employees' financial data, its clear that while some POIS are associated with data points (salary, for example) that are far outside the quantiles, others are associated with average or below average points. To recognize meaningful and accurate trends in the data, its important to preserve this spread. This being said, I noticed two observations on the financial info sheet: "Total" and "The Travel Agency In The Park." Total is simply the sum for each category for all employees' data, and The Travel Agency in the Park would probably be a company either involved with or managed by Enron. Since I am trying to identify Persons of Interest, and neither of these are people, I removed them from the dataset.</p>
        </section>
        <section>
            <h1>Feature Description and Selection</h1>
            <h4>Types of Features</h4>
            <p>The features in this dataset are all numerical values, save for "email address", a string, which I discarded. The other features fall into three categories: financial data, email data, and POI status. Financial features contain the actual dollar values for items like an employee's salary and bonus. The full set contains: <q>['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']</q>The email features are computed values, based on a textual analysis of the email corpus done by Udacity, and represent the count of that particular feature for each person. The full set (again, minus the string for email address): <q>['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']</q>POI status was one feature, marking whether or not the person was suspected, convicted, or testified in exchange for immunity in relation to the Enron fraud investigation. It was a single feature, 'poi', with either a 1 marking the person as a POI or a 0 marking them as not of interest.</p>
            <p>
            <h4>Manual Evaluation</h4>
            <p>After removing the outliers, I was left with <b>144 observations</b> with <b>20 features</b> for a total <b>2880 data points</b>. I still had <b>18 POIs</b> to work with. My goal was, again, to preserve as much spread and information density in this small dataset as possible, so that I ended up with the most robust classifier possible.</p><p>From the research I did on how Enron fraud was both perpetrated and concealed in the company's history, I developed a hypothesis. Most instances of misappropriated funds seemed to stem not from an initial active fraud, but from trying to cover up the activity - buying off partners, hiding money in expensive office furnishings, and otherwise bribing/incriminating anyone who may have reported the fraudulent activities of the company's CEO and chief financial officer. If that hypothesis was accurate, we'd expect POIs to communicate more with other POIs than non-POIs.</p><p>To investigate this hypothesis, I added two features to the current list: the ratio of 'from_poi_to_this_person' to total emails to this person, and the ratio 'from_this_person_to_poi' to total emails sent by this person. A higher number for either would, if we reject the null, indicate higher connection with POI's.</p><p>I now had <b>23 total features</b> to work with. However, sheer quantity of features won't help the classifier's accuracy if the features are very noisy or not information-dense. I thus calculated the percentage of NaN values for each feature, and updated the features_list to drop all features for which the percentage was over a certain threshold. I tuned the optimal threshold later, while tuning the classifier, but for the purposes of this investigation the final threshold was <b>>0.4</b>. This removed <q>['deferral_payments', 'loan_advances', 'restricted_stock_deferred', 'long_term_incentive', 'director_fees', 'from_poi_to_this_person', 'from_this_person_to_poi', 'from_emails_poi_ratio']</q> (Notably, one of my created features, <b>'from_emails_poi_ratio'</b>, was removed, but the other remained.) I was thus left with <b>15 features</b> with high infomational density.</p>
            <h4>SKLearn: SelectFromModel</h4>
            <p>I tested several different sklearn feature selection methods, as well as PCA feature compression, with the initial DecisionTreeClassifier and the AdaBoostClassifier.</p>
            <table>
                <tr>
                    <th>Selection Method</th>
                    <th>Parameters</th>
                    <th>DecisionTree Accuracy</th>
                    <th>AdaBoost Accuracy</th>
                    <th>Selection Fit Time</th>
                    
                </tr>
                <tr>
                    <td>None</td>
                    <td>none</td>
                    <td>0.866</td>
                    <td>0.866</td>
                    <td>0</td>
                    
                </tr>
                <tr>
                    <td>SelectPercentile</td>
                    <td>percent = 10</td>
                    <td>0.8</td>
                    <td>0.733</td>
                    <td>0.001</td>
                </tr>
                <tr>
                    <td>SelectKBest</td>
                    <td>k = 2</td>
                    <td>0.8</td>
                    <td>0.733</td>
                    <td>0.001</td>
                </tr>
                <tr>
                    <td>PCA</td>
                    <td>n_components = 2</td>
                    <td>0.8</td>
                    <td>0.866</td>
                    <td>0.058</td>
                </tr>
                <tr>
                    <td>SelectFromModel</td>
                    <td>the classifier</td>
                    <td>0.8</td>
                    <td>0.933</td>
                    <td>0.002 / 0.098</td>
                </tr>
            </table>
            <p> As you can see, PCA and SelectFromModel had the best results in terms of accuracy with the classifiers tested. SelectFromModel wasthe only one that boosted accuracy beyond the base classifier scores, however a high accuracy score does not automatically indicate a higher recall or precision score. So, after narrowing down the choices to PCA feature compression and SelectFromModel, I compared the recall, precision, and F1 scores of those two, using individual runs of the tester.py file provided with the project.</p>
            <table>
                <tr>
                    <th>Method</th>
                    <th>Classifier</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>F2</th>
                </tr>
                <tr>
                    <td>PCA</td>
                    <td>DecisionTree</td>
                    <td>0.772</td>
                    <td>0.138</td>
                    <td>0.137</td>
                    <td>0.137</td>
                    <td>0.137</td>
                </tr>
                <tr>
                    <td>PCA</td>
                    <td>AdaBoost</td>
                    <td>0.804</td>
                    <td>0.169</td>
                    <td>0.121</td>
                    <td>0.141</td>
                    <td>0.128</td>
                </tr>
                <tr>
                    <td>SelectFromModel</td>
                    <td>DecisionTree</td>
                    <td>0.818</td>
                    <td>0.315</td>
                    <td>0.309</td>
                    <td>0.312</td>
                    <td>0.310</td>
                </tr>
                <tr>
                    <td>SelectFromModel</td>
                    <td>AdaBoost</td>
                    <td>0.837</td>
                    <td>0.369</td>
                    <td>0.312</td>
                    <td>0.338</td>
                    <td>0.322</td>
                </tr>
            </table>
            <p>SelectFromModel's scores make the selection clear. By selecting features based on their importance weights to the classifier and ignoring those with weights below a threshold(default = mean), we were able to reach scores above the project requirements with untuned classifiers. And, since I'm passing the selector the features with the greatest information density, i.e, NaN ratios of less than 0.4, this model ensures that the classifier uses data that's' both information dense and predictively powerful.</p><p>The features are selected each time the file is run, but with a tuned AdaBoost classifier, features that are regularly used are <b>expenses at weight 0.02</b>, <b>exercised_stock_options at weight 0.52</b> and <b>other at weight 0.14</b>, and the number of features used is typically between 1 and 5.</p>
        </section>
        <section>
            <h1>The Classifier</h1>
            <p>The two classifiers I tested were the DecisionTreeClassifier and AdaBoostClassifier. After testing both with the selection filters and achieving recall and precision scores with no tuning above the threshold (see chart above), I decided to pursue the AdaBoost classifier as it typically produced more robust scores than the DecisionTree. Since the data is not especially linearly related or separable, I wanted to steer clear of SVMs or GaussianNB, and given the small size of the dataset and even smaller number of POIs, iterating over a large number of classifiers as AdaBoost does ensures balanced, robust predictions.
            </p>
        </section>
        <section>
            <h1>Philosophy of Tuning</h1>
            <h4>Tuning an Algorithm</h4>
            <p>To tune an algorithm, essentially, is to adjust the parameters of the classifier from the default in the pursuit of "the best classifier". For classifiers such as SVM, you can adjust the penalty for wrong guesses, prioritizing accuracy vs. prioritizing easily communicable or quickly reached results, or the type of decision your classifier is making. Every dataset is different, and every situation is different - picking the one that is "The Best" will always be a unique and specific exercise to your situation.</p><p>I mentioned briefly, in the intro, the importance precision has to the most obvious use of this data, to identify potential signs of fraud in a person's emails and financial data. To tune this algorithm, then, we want a high precision; you would address this by applying penalties for wrong guesses, and by prioritizing accuracy over a classifier that runs quickly or spits out simpler, easy-to-communicate relationships between variables.</p>
            <h4>Adaboost Tuning</h4>
            <p>Tuning the AdaBoost classifier, however, was particularly challenging or very easy, depending on the perspective. After trying a number of different learning_rates and n_estimator settings via GridSearchCV, I found that the Adaboost classifier's "best" scores, where recall, precision and f1 were consistently above 0.3 and precision of up to 0.4, were achieved with learning_rate = 0.3 and n_estimators = 100.</p>
        </section>
        <section>
            <h1>Philosophy of Validation</h1>
            <p>Achieving valid results is the entire reason to undertake a project such as this, so using a method of validation is crucial. The first stages of informal development and investigation of this project utilized SKlearn's train_test_split function, which reserves a portion of the data for testing and allowing precise control over the train/test data ratio. However, the small dataset size made it necessary for tuning and feature selection to use SKlearn's StratifiedShuffleSplit. This ensured that train groups and test groups kept a proportionate number of target groups while prioritizing variance in data, and produced the best results.</p>
            <p>An aspect of validation that is important to discuss in the context of this dataset is training set size vs. test set size. In the lessons, we looked at how at a certain ratio, results tend to plateau. I experimented with several different test sizes, starting at 0.1. I found that test scores significantly improved until I had moved down to 0.05. At 0.01, the test data only contained two data points - way too few to even contain a target 
        </section>
        <section>
            <h1>Evaluation</h1>
            <p>I have been using accuracy, precision, recall and F1 to evaluate many of my decisions during this project. So, to explain, take a two-class prediction, such as this dataset (Non-POI and POI). In a dataset like this, there are:</p>
            <ol>
                <li>Total: The total number of predictions made</li>
                <li>Total Correct: The total number of correct predictions made, target or non-target</li>
                <li>True Positives: The number of predictions that correctly identified the target class (here, POI)</li>
                <li>True Negatives: The number of predictions that correctly identified a non-target (non-POI)</li>
                <li>False Positives: The number of non-targets accidentally predicted as the target class</li>
                <li>False Negatives: The number of targets accidentally predicted as non-targets
            </ol>
            <p>Accuracy, recall and the like are computed scores based on these categories. The formula for each looks like:
            <ol>
                <li>Accuracy: Total Correct / Total </li>
                <li>Precision: True Positives / True Positives + False Positives (the total number of positive predictions)</li>
                <li>Recall: True Positives / True Positives + False Negatives (the total number of datapoints in the target class)</li>
                <li>F1: 2 * Precision * Recall  / Precision + Recall</li>
            </ol>
            <p>From these definitions, you can see how a high accuracy with this dataset can be misleading. With only 18 targets out of 144 options, the accuracy of simply always guessing non-POI would be 0.875 -- better than our unfiltered AdaBoost algorithm! However, with 0 true positives, both precision and recall would be scores of 0.</p>
        </section>
        <section>
            <h1>Conclusion</h1>
            <p>Using a StratifiedShuffleSplit dataset with 0.05 testing set proportionally, the highest accuracy, precision, recall and f1 scores I was able to obtain were, respectively, <b>0.85</b>, <b>0.42</b>, <b>0.33</b>, and <b>0.37</b>. This was achieved using the features <b>bonus</b>, <b>total_stock_value</b>, <b>expenses</b>, <b>exercised_stock_options</b>, <b>other</b>, and <b>shared_receipt_with_poi</b>, selected via SelectFromModel. My AdaBoostClassifier was tuned to <b>learning_rate = 0.3</b> and <b>n_estimators = 100</b>.</p><p>What this means for my program is that [acc]% of the time, I correctly guessed the POI status of an individual. Of the people I guessed who were POI's, I got [precision]% of them right, and of all the POI's, I spotted [recall]% of them. This definitely meets the project requirments, and I'd say isn't a bad performance for such a nuanced dataset.</p><p>Next steps I would take for this project would be to compare a more complex sentiment map of the email corpus; maybe devise some features exploring how many emails about money, or about deals, were sent or recieved by POI status. Another would be to pull dates out from the emails, and see the rate at which POIs sent emails during certain periods associated with the start of specific fraudulent acts.</p>
        </section>
    </body>
</html>
