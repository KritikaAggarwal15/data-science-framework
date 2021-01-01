import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython.display import display


class eda:
    
    def __init__(self): 
        self.geek = "GeekforGeeks"

    def read_file(self,file):
        #os.chdir('C:\\Users\KRITIKA AGARWAL\\Documents\\Data Science\\Framework\\data')
        os.chdir('../data')
        
        if file.endswith('.xlsx'):
            dataframe = pd.read_excel(file)
            
        elif file.endswith('.json'):
            dataframe = pd.read_json(file)
            
        elif file.endswith('.csv'):
            dataframe = pd.read_csv(file)
        
        else:
            dataframe = pd.read_excel(file)
        
        os.chdir('../case_studies')
        
        return dataframe
    
    def export_to_excel(self,dataframe):
        #os.chdir('C:\\Users\KRITIKA AGARWAL\\Documents\\Data Science\\Framework\\output')
        os.chdir('../output')
        dataframe.to_excel('combined_classification_report.xlsx')
        os.chdir('../case_studies')
        return

    def export_to_csv(self,dataframe,name):
        dataframe.to_csv(name)
        return

    def describe(self,dataframe,columns = []):
        display(dataframe[columns].describe())
        return

    def drop_columns_by_colnums(self,dataframe,col_num = []):
      
        dataframe=dataframe.drop(dataframe.columns[col_num], axis=1)
        return dataframe
    
    def drop_columns_by_colnames(self,dataframe,col_name = []):
        dataframe=dataframe.drop(col_name, axis=1)
        return dataframe

    def get_categorical_cols(self,dataframe):
        categorical_columns=[]
        for column in dataframe.columns:
            if dataframe[column].dtype == 'object':
                categorical_columns.append(column)
        print("Categorical Columns are :",categorical_columns)
        return categorical_columns
    
    def get_continous_cols(self,dataframe):
        continous_columns=[]
        for column in dataframe.columns:
            if dataframe[column].dtype == 'int64' or dataframe[column].dtype == 'float64':
                continous_columns.append(column)
        return continous_columns


    def info(self,dataframe):
        display("Dataset Information")
        display(dataframe.info())
        return 

    def shape(self,dataframe):
        display("Dataset Shape")
        display(dataframe.shape)
        return
    
    def check_null_values(self,dataframe):
        display("Check for Null or missing values")
        display(dataframe.isnull().sum())
        return

    def replace_null_values_with_mean(self,dataframe):
        display("Fill Null or missing values with mean")
        dataframe.fillna(dataframe.mean(),inplace=True)
        return

    def replace_null_values_with_mode(self,dataframe):
        display("Check for Null or missing values with Mode")
        dataframe.fillna(dataframe.mode(),inplace=True)
        return
    
    def replace_null_values_with_mediam(self,dataframe):
        display("Check for Null or missing values with Median")
        dataframe.fillna(dataframe.median(),inplace=True)
        return
    
    def check_duplicate_data(self,dataframe):
        display("Check for Duplicate Data")
        duplicates = dataframe.duplicated()
        print('Number of duplicate rows = %d' % (duplicates.sum()))
        print(dataframe.shape)
        return dataframe[duplicates]
    
    def drop_duplicate_data(self,dataframe):
        print('Dataset Before Dropping duplicate Data',dataframe.shape)
        dataframe.drop_duplicates(inplace=True) 
        print('Dataset After Dropping Duplicate Data',dataframe.shape)
        return 
    

    def check_unique_value_categorical_var(self,dataframe):
        display("Check unique values for categorical variables")
        for column in dataframe.columns:
            print(column.upper(),': ',dataframe[column].nunique())
            print(column.upper(),': ',dataframe[column].unique())
            print(dataframe[column].value_counts().sort_values())
            print('\n')
        return 

    def check_junk_value_cont_var(self,dataframe):
        display("Check Junk values for continous variables")
        junk_data=dataframe[~dataframe.applymap(np.isreal).all(1)]
        print(junk_data)
        if (len(junk_data)==0):
            display("There is no junk Value")
        else:
            return junk_data

    def create_boxplot(self,dataframe):
        plt.subplots(figsize=(20,7))
        sns.boxplot(data=dataframe,orient ='h')
        plt.grid()
        #plt.savefig('BoxPlot_1', bbox_inches='tight', pad_inches=0.05)
        plt.show()
        return

    def univariate_analysis(self,dataframe,cols= []):
        fig,axes = plt.subplots(nrows=len(cols),ncols=2)
        fig.set_size_inches(17,24)
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
        return
    
    def remove_outliers(self,dataframe,cols= []):
        sorted(dataframe)
        for column in dataframe[cols].columns:
            Q1,Q3=np.percentile(dataframe[column],[25,75])
            IQR=Q3-Q1
            lr= Q1-(1.5 * IQR)
            ur= Q3+(1.5 * IQR)
            dataframe[column]=np.where(dataframe[column]>ur,ur,dataframe[column])
            dataframe[column]=np.where(dataframe[column]<lr,lr,dataframe[column])
        return 

    def convert_cat_into_code(self,dataframe,cols= []):
        
        for column in dataframe[cols].columns: 
            print ("Columns changed into code",column)
            dataframe[column] = pd.Categorical(dataframe[column]).codes
        return dataframe


    def change_dtype(self,dataframe,dtype_data):
        for i in dtype_data.index:
            print(dtype_data.iloc[i][0])
            dataframe[dtype_data.iloc[i][0]]=dataframe[dtype_data.iloc[i][0]].astype(dtype_data.iloc[i][1])
        return 
   

    def create_dataframe(self,data,cols):
        dataframe = pd.DataFrame(data,columns=cols)
        return dataframe

    def replace_NaN_with_None(self,dataframe):
        dataframe.replace({np.NAN: None},inplace=True)
        return dataframe
