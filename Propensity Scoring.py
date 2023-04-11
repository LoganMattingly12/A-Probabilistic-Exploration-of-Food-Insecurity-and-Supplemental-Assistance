"""
Created on Mon Apr 10 19:48:25 2023

@author: Logan
"""
import pandas as pd
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *


df = pd.read_excel('C:\\Users\Logan\Dropbox\Thesis - new\Data\propensity score vars.xlsx')
df=df.dropna()
df['benefits'] = df['benefits'].apply(
    lambda x: 1 if x > 0  else (0 if x == 0 else None))


psm = PsmPy(df, treatment='benefits', indx='household_num', exclude = [])
psm.logistic_ps(balance=True)
#psm.predicted
psm.knn_matched(matcher='propensity_score', replacement=False, caliper=None, drop_unmatched=True)

#psm.plot_match(Title='Test', Ylabel='Household number', Xlabel= 'Propensity logit', names = ['treatment', 'control'], save=True)

psm.effect_size_plot(save=False)
