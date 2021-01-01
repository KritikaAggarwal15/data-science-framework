import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython.display import display

class univariate_analysis:

    def univariate_analysis(self,dataframe,cols= []):
        fig,axes = plt.subplots(nrows=len(cols),ncols=2)
        fig.set_size_inches(10,24)
        text1="Distribution "
        text2="Boxplot "
        i=0
        for column in cols:
            a = sns.distplot(dataframe[column] , ax=axes[i][0])
            label=column+ " " +text1
            a.set_title(label,fontsize=10)
            a = sns.boxplot(dataframe[column] , orient = "v" , ax=axes[i][1],)
            label=column + " "+text2
            a.set_title(label,fontsize=10)
            i=i+1
        plt.tight_layout()
        plt.show()