# EnronDataAnalysis
        <h1>Enron Data Analysis with AdaBoost and SelectFromModel</h1>
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
        </section>
            <h1>Conclusion</h1>
            <p>Using a StratifiedShuffleSplit dataset with 0.05 testing set proportionally, the highest accuracy, precision, recall and f1 scores I was able to obtain were, respectively, <b>0.85</b>, <b>0.42</b>, <b>0.33</b>, and <b>0.37</b>. This was achieved using the features <b>bonus</b>, <b>total_stock_value</b>, <b>expenses</b>, <b>exercised_stock_options</b>, <b>other</b>, and <b>shared_receipt_with_poi</b>, selected via SelectFromModel. My AdaBoostClassifier was tuned to <b>learning_rate = 0.3</b> and <b>n_estimators = 100</b>.</p><p>What this means for my program is that [acc]% of the time, I correctly guessed the POI status of an individual. Of the people I guessed who were POI's, I got [precision]% of them right, and of all the POI's, I spotted [recall]% of them. This definitely meets the project requirments, and I'd say isn't a bad performance for such a nuanced dataset.</p><p>Next steps I would take for this project would be to compare a more complex sentiment map of the email corpus; maybe devise some features exploring how many emails about money, or about deals, were sent or recieved by POI status. Another would be to pull dates out from the emails, and see the rate at which POIs sent emails during certain periods associated with the start of specific fraudulent acts.</p>
        </section>
    </body>

