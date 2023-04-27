# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:23:25 2022

@author: ltmat
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import susi
from susi.SOMPlots import plot_nbh_dist_weight_matrix, plot_umatrix

X = pd.read_excel('C:\\Users\ltmat\Documents\Logan\School\MSU\Thesis\Data\ETL Data\Demographics 1.xlsx')

rows = 50
columns = 50

som = susi.SOMClustering(n_rows = rows, n_columns=columns)
som.fit(X)
print("SOM Fitted")

u_matrix = som.get_u_matrix()
plot_umatrix(u_matrix, rows, columns,cmap='summer')
plt.show()

plot_nbh_dist_weight_matrix(som)
plt.show()
