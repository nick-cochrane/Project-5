# Recession Probability Modeling

Project developed in Python to determine the probability of recession 6-12 months from now. 
* Classification features used throughout the project include the 10 year Treasury yield - 3 month T-Bill yield, consumer sentiment index, and copper/gold price ratio. 
* A baseline logistic regression model was established and influenced by the New York Federal Reserve's widely followed recession probability model. The baseline logistic regression model used the 10 Year - 3 Month yield as its only feature. 
* A suite of classification models - logistic regression, Random Forest, XGBoost, and Naive Bayes - were tested and scored based on AUC. 

Naive Bayes outperformed all models with an AUC of 0.83. Please see below for the current chance of recession 6-12 months from now.


![Recession Prob Graph](https://github.com/nick-cochrane/Recession-Modeling/blob/master/Data/prob_rec.png)
