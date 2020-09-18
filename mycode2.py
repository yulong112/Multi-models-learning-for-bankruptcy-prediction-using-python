# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 13:17:40 2020

@author: 60342
"""


# In[1]: Import several important libs.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn import metrics
from sklearn.metrics import confusion_matrix
get_ipython().magic('matplotlib inline')

# In[2]: Function definition used for data process and model training.

'''Function of splitting the data to features and the labels'''
def preprocessdata(raw_data):
#    labels_bankruptcy_flag, bankruptcy_factors = dmatrices('class ~ trans_cf_td + trans_ca_cl + trans_re_ta + trans_ni_ta + trans_td_ta + trans_s_ta + trans_wc_ta + trans_wc_s + trans_c_cl + trans_cl_e + trans_in_s + trans_mve_td',
#                      raw_data, return_type="dataframe")
    labels_bankruptcy_flag=raw_data['class']
    labels_bankruptcy_flag=np.array(labels_bankruptcy_flag,dtype=float)
#        labels_bankruptcy_flag = np.ravel(labels_bankruptcy_flag)

    bankruptcy_factors=raw_data.copy()
    bankruptcy_factors=bankruptcy_factors.drop(['class'],axis=1)
#    bankruptcy_factors=bankruptcy_factors.drop(['ID'],axis=1)
#    
    return labels_bankruptcy_flag,bankruptcy_factors

'''Function of calculating performance indexes'''
def performance_indexes(true_labels,predicted_labels, predicted_proba=[]):
    print (metrics.accuracy_score(true_labels,predicted_labels))
    if len(predicted_proba):
        print (metrics.roc_auc_score(true_labels, predicted_proba[:, 1]))
    
    print (metrics.confusion_matrix(true_labels,predicted_labels))
    print (metrics.classification_report(true_labels,predicted_labels))
    cal_confusion_mat = confusion_matrix(true_labels,predicted_labels)
    plt.figure(figsize=(10,6))
    sns.heatmap(cal_confusion_mat,  
                xticklabels=['Non Bankrupt', 'Bankrupt'], 
                yticklabels=['Non Bankrupt', 'Bankrupt'])
    plt.show()
    return cal_confusion_mat

'''Function of training bankruptcy model'''
def train_bankruptcy_model(training_data,select_model):
    
    '''2_1.split the training data to features and the labels'''
    train_label_bankruptcy_flag,training_bankruptcy_factors=preprocessdata(training_data)
    print (training_bankruptcy_factors.columns)
    
    '''2_2.build the selected machine learning model'''
    if select_model=='LR':
        # Logistic Regression
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif select_model=='Dtree':
        # Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
    elif select_model=='MLP':
        # MLP Neural Network
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=(12,12,12))
    elif select_model=='SVM':
        # Support Vector Machine
        from sklearn.svm import SVC
        model = SVC(probability = True)

    '''2_3.training model'''
    model = model.fit(training_bankruptcy_factors, train_label_bankruptcy_flag)
    # check the accuracy on the training set
    acc=model.score(training_bankruptcy_factors, train_label_bankruptcy_flag)
    print('Evaluation of ',select_model,' model using the training data: ',acc)

#    print('Percentage of bankruptcy on training data：',train_label_bankruptcy_flag.mean())

    ############################## analysis and results ###################################
    
    '''2_4.predict labels of training data using model'''
    predicted_train_labels = model.predict(training_bankruptcy_factors)
#    print (predicted_train_labels)
    
    '''2_5.probabilities of classification by model'''
    proba_training = model.predict_proba(training_bankruptcy_factors)
#    print (proba_training)
    
    '''2_6.calculate score, confusion matrix and other performance indexes'''
    train_confusion_mat=performance_indexes(train_label_bankruptcy_flag, predicted_train_labels, proba_training)
    
    '''2_7.calculate VIF'''
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = [variance_inflation_factor(training_bankruptcy_factors.values, i) for i in range(training_bankruptcy_factors.shape[1])]
#    print(vif)
    
    return model,vif
  
'''Function of training bankruptcy model'''
def predict_bankruptcy_result(test_label_bankruptcy_flag,test_bankruptcy_factors,select_model,bankruptcy_model):
    
    ############################## analysis and results ###################################
    
    '''2_1.predict labels of testing data using model'''
    predicted_test_labels = bankruptcy_model.predict(test_bankruptcy_factors)
    print (predicted_test_labels)
    
    acc=bankruptcy_model.score(test_bankruptcy_factors, test_label_bankruptcy_flag)
    print('Evaluation of ',select_model,' model on testing bankruptcy data: ',acc)
    
    print('Percentage of bankruptcy on testing bankruptcy data：',test_label_bankruptcy_flag.mean())
    
    '''2_2.probabilities of classification by model'''
    proba_testing = bankruptcy_model.predict_proba(test_bankruptcy_factors)
    print (proba_testing)
    
    '''2_3.calculate score, confusion matrix and other performance indexes'''
    test_confusion_mat=performance_indexes(test_label_bankruptcy_flag, predicted_test_labels, proba_testing)
    
    predicted_test_labels = pd.Series(predicted_test_labels)

    return predicted_test_labels,proba_testing

# In[3]: Classification main function with training and testing.
'''load data and preprocess'''
from scipy.io import arff
select_data="1year.arff"
All_bankruptcy_data,meta=arff.loadarff(select_data)
All_bankruptcy_data=pd.DataFrame(All_bankruptcy_data)
All_bankruptcy_data['class']=All_bankruptcy_data['class'].apply(lambda row_x: int(bytes.decode(row_x)))

All_bankruptcy_data = All_bankruptcy_data.drop(columns=['Attr37', 'Attr21'])
All_bankruptcy_data.fillna(0, inplace=True)

'''3_1.select the training data'''
training_bankruptcy_data = All_bankruptcy_data.sample(frac=0.5, random_state=0)

'''3_2.plot the bar graph reflecting the count of two labels -- bankruptcy or not'''
plt.figure(figsize=(10,6))
sns.countplot(x='class',data = training_bankruptcy_data)
plt.show()

'''3_3.load the testing data'''
testing_bankruptcy_data=All_bankruptcy_data.loc[~All_bankruptcy_data.index.isin(training_bankruptcy_data.index)]
testing_bankruptcy_data.head()
'''3_4.plot the bar graph reflecting the count of two labels -- bankruptcy or not'''
plt.figure(figsize=(10,6))
sns.countplot(x='class',data = testing_bankruptcy_data)
plt.show()
'''3_5.split the testing data to features and the labels'''
test_label_bankruptcy_flag,test_bankruptcy_factors=preprocessdata(testing_bankruptcy_data)
#    y_test, X_test = dmatrices('class ~ trans_cf_td + trans_ca_cl + trans_re_ta + trans_ni_ta + trans_td_ta + trans_s_ta + trans_wc_ta + trans_wc_s + trans_c_cl + trans_cl_e + trans_in_s + trans_mve_td',
#                      test_data, return_type="dataframe")
    
   
model_name_all=['LR','Dtree','MLP','SVM']
composite_predlabels = pd.DataFrame()
#select_model='LR'
for select_model in model_name_all:
    print('------Using ',select_model,' model for training------')
    '''3_5.training the bankruptcy model'''
    bankruptcy_model,bankruptcy_VIF=train_bankruptcy_model(training_bankruptcy_data,select_model)
    
    '''3_6.testing the bankruptcy testing data'''
    predicted_test_labels,proba_testing=predict_bankruptcy_result(test_label_bankruptcy_flag,test_bankruptcy_factors,select_model,bankruptcy_model)
    
    '''3_7.generate composite predictive labels'''
    composite_predlabels[select_model] = predicted_test_labels

#print (composite_predlabels)
composite_predicted_bankrupt = composite_predlabels[['LR','MLP','Dtree']].mode(axis=1,numeric_only=True)
#print(composite_predicted_bankrupt)
print (metrics.accuracy_score(test_label_bankruptcy_flag, composite_predicted_bankrupt))

'''3_8.calculate score, confusion matrix and other performance indexes'''
final_test_confusion_mat=performance_indexes(test_label_bankruptcy_flag, composite_predicted_bankrupt)
    
    



