
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score


import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns

def feature_dashboard(date_series, rec_series, feature_series, title = '', date_label = 'Time', feature_label = '', 
					  feature_color = 'blue', horizontal_line = False):
	"""
	Function that outputs a recession dashboard: three time series plots of recession vs. respective features

	Paramaters:
	--------------
	date_series: range of dates for x-axis of dashboard
	rec_series: binary series of recessions, as classified by NBER
    feature_series: feature series to be plotted against time
    date_label: x-axis title
    feature_label: y-axis title
    feature_color: feature line color
    horizontal_line: 'False', if true then horizontal line at origin is displayed
	"""

	fig, ax1 = plt.subplots(figsize=(20,10))

	color = 'gray'
	ax1.set_xlabel(date_label, fontsize = 30)
	ax1.set_ylabel('Recession', color=color, fontsize = 30)
	ax1.plot(date_series, rec_series, color=color, label = 'Shaded Gray Indicates Recession')
	ax1.fill_between(date_series, 0, rec_series, alpha = 0.4, color = 'gray')
	ax1.set_yticks(np.arange(0.01, 1, step=1))
	ax1.set_ylim([0.01,0.5])
	ax1.tick_params(axis='y', labelcolor=color, labelsize = 0)
	ax1.tick_params(axis='x', labelsize = 20)
	fig.suptitle(title, y = 1.05, fontsize=40)
	plt.legend(loc='upper right', fontsize = 15)

	ax2 = ax1.twinx()  

	color = feature_color
	ax2.set_ylabel(feature_label, color=color, fontsize = 30)  
	ax2.plot(date_series, feature_series, color=color)
	ax2.tick_params(axis='y', labelcolor=color, labelsize = 20)

	fig.tight_layout()
	if horizontal_line == True:
		plt.axhline(0, color="black")
	plt.show();


def rolling_6(y_test):
    """
    Helper function to convert y_test to rolling 6 month window. 
    Returns numpy array: If recession occurs in 6 month window from index, returns 1. Otherwise, returns 0
    """
    rec_6_months = []
    
    for i in range(len(y_test)):
        if np.sum(y_test[i:i+6]) > 0:
            rec_6_months.append(1)    
        
        else:
            rec_6_months.append(0)
                
    return np.array(rec_6_months)

def fit_score_viz(model, X_train, y_train, X_test, y_test, already_fit = False):
 	if already_fit == True:
 		fit_model = model
 		fit_score(fit_model, X_train, y_train, X_test, y_test)
 	else:
 		fit_model = fit_score(model, X_train, y_train, X_test, y_test)

 	make_confusion_matrix(fit_model, X_test, y_test)
 	prec_recall(model, X_test, y_test)

 	return fit_model

def fit_score(model, X_train, y_train, X_test, y_test):
	fit_model = model.fit(X_train, y_train)
	y_predict = fit_model.predict(X_test)
	y_probs_1 = fit_model.predict_proba(X_test)
	print("Test set AUC: {:6.2f}%".format(roc_auc_score(y_test, y_probs_1[:,1])))
	print("Default Threshold: 0.5")
	print("Test Set Precision: {:6.4f}".format(precision_score(y_test, y_predict)))
	print('\n')

	return fit_model

def make_confusion_matrix(model, X_test, y_test, threshold=0.2):
    # Predict class 1 if probability of being in class 1 is greater than threshold
    # (model.predict(X_test) does this automatically with a threshold of 0.5)
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    default_confusion = confusion_matrix(y_test, y_predict)
    plt.figure(dpi=80)
    sns.heatmap(default_confusion, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
           xticklabels=['Growth', 'Recession'],
           yticklabels=['Growth', 'Recession']);
    plt.xlabel('prediction')
    plt.ylabel('actual')
    print("Scores for threshold: " + str(threshold))
    print("F1: {:6.4f}, Precision: {:6.4f},   Recall: {:6.4f}".format(fbeta_score(y_test, y_predict, beta = 1),
                                                                           precision_score(y_test, y_predict), 
                                                                           recall_score(y_test, y_predict)))

def prec_recall(model, X_test, y_test):
	precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
	plt.figure(dpi=80)
	plt.plot(threshold_curve, precision_curve[1:],label='precision')
	plt.plot(threshold_curve, recall_curve[1:], label='recall')
	plt.legend(loc='lower left')
	plt.xlabel('Threshold (above this probability, label as default)');
	plt.title('Precision and Recall Curves')

	fig_1 = plt.gcf()
	fig_1.savefig('Data/logit_curves_1', format='png')
    