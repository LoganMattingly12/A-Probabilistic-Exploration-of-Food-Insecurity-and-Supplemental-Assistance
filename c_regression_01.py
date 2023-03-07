# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:33:07 2023

@author: ltmat
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.read_excel('C:\\Users\ltmat\Documents\Logan\School\MSU\Thesis\Data\Household_DataSO_withtotalexp_demreduction.xlsx')
scaler = StandardScaler()
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


print('XGBoost')
test2 = (accuracy_score(y_test,y_hat))
auc = metrics.roc_auc_score(y_test,y_hat)
print('The accuracy of Testing is '+str(test2))
print('\n')
print(classification_report(y_test, y_hat))
print('The AUC is '+str(auc))
cm = confusion_matrix(y_test,y_hat)
print(cm)



# plot feature importance
plot_importance(xgb)
plt.show()