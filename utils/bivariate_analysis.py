import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython.display import display


class bivariate_analysis:

    def corr_coef(self,dataframe):
        print(dataframe.corr(method='pearson'))
        return

    def plot_corr_coef_heatMap(self,dataframe):
        plt.subplots(figsize=(10,5))
        total_columns=len(dataframe.columns)
        sns.heatmap(dataframe.corr(), annot=True,fmt='.2f',mask=np.triu(dataframe.iloc[:,0:total_columns].corr(),+1))
        #plt.savefig('HeatMap', bbox_inches='tight', pad_inches=0.05)
        plt.show()
        return

    def plot_corr_coef_pairPlot(self,dataframe,target_column):
        #sns.set(style="ticks", color_codes=True)
        sns.pairplot(dataframe , hue=target_column , diag_kind='hist')
        plt.show()
        return