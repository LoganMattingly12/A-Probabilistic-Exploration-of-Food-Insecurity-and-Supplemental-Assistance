# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:33:07 2023

@author: ltmat
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from xgboost import plot_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *



df = pd.read_excel('C:\\Users\Logan\Dropbox\Thesis - new\Data\Household_DataSO_withtotalexp_demreduction.xlsx')
scaler = MinMaxScaler()
df=df.dropna()
df['benefits'] = df['benefits'].apply(
    lambda x: 1 if x > 0  else (0 if x == 0 else None))


y = df['benefits']
df.drop(columns=['benefits'],inplace=True)
x = scaler.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1234)


xgb = xgb.XGBClassifier()
xgb.fit(X_train,y_train)
y_hat = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)


logr = linear_model.LogisticRegression(random_state=1234, max_iter=100)
logr.fit(X_train,y_train) 
y_pred_logr = logr.predict(X_test)

rf = RandomForestClassifier(n_estimators=1000).fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)

print('XGBoost')
test2 = (accuracy_score(y_test,y_hat))
auc = metrics.roc_auc_score(y_test,y_hat)
print('The accuracy of Testing is '+str(test2))
print('\n')
print(classification_report(y_test, y_hat))
print('The AUC is '+str(auc))
cm = confusion_matrix(y_test,y_hat)
print(cm)


print('Logistic Regression')
test2 = (accuracy_score(y_test,y_pred_logr)*100)
y_pred_proba = logr.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print('The accuracy of Testing is '+str(test2))
print('\n')
print(classification_report(y_test, y_pred_logr))
print('The AUC is '+str(auc))
cm = confusion_matrix(y_test,y_pred_logr)
print(cm)

print('Random Forest')
test2 = (accuracy_score(y_test,y_pred_rf)*100)
auc = metrics.roc_auc_score(y_test, y_pred_rf)
print('The accuracy of Testing is '+str(test2))
print('\n')
print(classification_report(y_test, y_pred_rf))
print('The AUC is '+str(auc))
cm = confusion_matrix(y_test,y_pred_rf)
print(cm)


# plot feature importance
#plot_importance(xgb)
#plt.show()

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_logr)
fpr1,tpr1, _ = metrics.roc_curve(y_test, y_hat)
fpr2,tpr2, _ = metrics.roc_curve(y_test,y_pred_rf)
plt.plot(fpr,tpr)
plt.plot(fpr1,tpr1)
plt.plot(fpr2,tpr2)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
