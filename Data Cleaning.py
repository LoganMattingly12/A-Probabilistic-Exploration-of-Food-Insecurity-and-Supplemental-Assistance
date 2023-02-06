# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:54:59 2022

@author: ltmat
"""
import pandas as pd

X = pd.read_excel('C:\\Users\\faps_household_remanedxlsx.xlsx')
data = pd.DataFrame()
data = X.iloc[:,0:260].replace(to_replace=(-996,-997,-998,-999,'.'),value=(0,0,0,0,0))

def missing(dff):
    print (round((dff.isnull().sum() * 100/ len(dff)),2).sort_values(ascending=False))
missing(data)

def rmissingvaluecol(dff,threshold):
    l = []
    l = list(dff.drop(dff.loc[:,list((100*(dff.isnull().sum()/len(dff.index))>=threshold))].columns, 1).columns.values)
    print("# Columns having more than %s percent missing values:"%threshold,(dff.shape[1] - len(l)))
    print("Columns:\n",list(set(list((dff.columns.values))) - set(l)))
    return l
l = rmissingvaluecol(data, 1)
data = data[l]

data.to_excel(r'C:\\Users\\ltmat\\Documents\\Logan\\School\\MSU\\Thesis\\Data\\faps_household_remaned_cleaned_1.xlsx')
