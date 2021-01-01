# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import numpy as np
import sys
import os
import traceback
from IPython.display import display
from utils import utils
from modals import modals
from bivariate_analysis import bivariate_analysis
import importlib, importlib.util

# %%
# Refer: https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
# Refer: https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# %%
util_eda = module_from_file("eda", "../utils/eda.py")
eda=util_eda.eda()

util_ba = module_from_file("bivariate_analysis", "../utils/bivariate_analysis.py")
bivariate_analysis=util_ba.bivariate_analysis()

util_un = module_from_file("univariate_analysis", "../utils/univariate_analysis.py")
univariate_analysis=util_un.univariate_analysis()

util_model = module_from_file("modals", "../utils/modals.py")
modals=util_model.modals()

# %%
model_data=eda.read_file("model_data.xlsx")
display(model_data)
# %%


# %%
sheet_name="Holiday_Package.csv"
target_variable='Holliday_Package'
modals_data=p.read_file("model_data.xlsx")
# %%
print(modals_data)
# %%
p.replace_NaN_with_None(modals_data)
modals_data
# %%
print(modals_data)

# %%
df_holiday_package=p.read_file(sheet_name)
display(df_holiday_package)

# %%
df_holiday_package=utils.drop_columns_by_colnums(df_holiday_package,[0])
display(df_holiday_package)

# %%
utils.info(df_holiday_package)

# %%
categorical_columns=utils.get_categorical_cols(df_holiday_package)
print(categorical_columns)

# %%
continuos_columns=utils.get_continous_cols(df_holiday_package)
print(continuos_columns)

# %%
utils.convert_cat_into_code(df_holiday_package,categorical_columns)
display(df_holiday_package)

# %%
utils.info(df_holiday_package)

# %%
cols_data=[['educ','int8'],['no_young_children','int8'],['no_older_children','int8']]
cols=['column','type']
dtype_data=utils.create_dataframe(cols_data,cols)
display(dtype_data)


# %%
utils.change_dtype(df_holiday_package,dtype_data)


# %%
utils.info(df_holiday_package)

# %%
continuos_columns=utils.get_continous_cols(df_holiday_package)
print(continuos_columns)

# %%
utils.describe(df_holiday_package,continuos_columns)

# %%
utils.describe(df_holiday_package,categorical_columns)

# %%
utils.shape(df_holiday_package)
# %%
utils.check_null_values(df_holiday_package)


# %%
utils.check_junk_value_cont_var(df_holiday_package[continuos_columns])

# %%
utils.check_duplicate_data(df_holiday_package)
# %%

bivariate_analysis.corr_coef(df_holiday_package[continuos_columns])


# %%
bivariate_analysis.plot_corr_coef_heatMap(df_holiday_package[continuos_columns])

# %%
bivariate_analysis.plot_corr_coef_pairPlot(df_holiday_package,target_variable)


# %%
utils.univariate_analysis(df_holiday_package,continuos_columns)


# %%

utils.remove_outliers(df_holiday_package,continuos_columns)

# %%
utils.create_boxplot(df_holiday_package[continuos_columns])
# %%

X,Y=modals.extract_target_column(df_holiday_package,target_variable)
# %%
print("X",X)

#%%
print ("Y",Y)

# %%
X_train,X_test,Y_train,Y_test = modals.split_data_into_train_test(X,Y,0.30,1)

# %%
print(X_train)

# %%
modals.check_imbalance(Y_train)

# %%
modals.check_imbalance(Y_test)

# %%
modals.check_dimension_train_test(X_train,X_test,Y_train,Y_test)
# %%
matrix=modals.create_comparision_matrix()

# %%
for i in modals_data.index:
    if(modals_data.iloc[i][0]=="ON"):
        X_train,X_test,Y_train,Y_test = modals.split_data_into_train_test(X,Y,0.30,1)
        best_grid,X_train,X_test=modals.build_modal(modals_data.iloc[i],X_train,Y_train,X_test)
        Y_train_predict,Y_test_predict=modals.get_predicted_target_variable(best_grid,X_train,X_test)
        Y_train_predict_prob,Y_test_predict_prob=modals.get_predicted_target_variable_prob(best_grid,X_train,X_test)
        train_auc=modals.get_auc_roc(best_grid,X_train,Y_train)
        print(train_auc)
        test_auc=modals.get_auc_roc(best_grid,X_test,Y_test)
        train_accuracy=modals.get_accuracy(best_grid,X_train,Y_train)
        print(train_accuracy)
        test_accuracy=modals.get_accuracy(best_grid,X_test,Y_test)
        print(test_accuracy)
        train_precsion,train_recall,train_f1=modals.get_classification_report(Y_train,Y_train_predict)
        modals.get_confusion_matrix(Y_train,Y_train_predict)
        test_precsion,test_recall,test_f1=modals.get_classification_report(Y_test,Y_test_predict)
        modals.get_confusion_matrix(Y_test,Y_test_predict)
        matrix[modals_data.iloc[i][5]+'_Train']=[train_accuracy,train_auc,train_recall,train_precsion,train_f1]
        matrix[modals_data.iloc[i][5]+'_Test']=[test_accuracy,test_auc,test_recall,test_precsion,test_f1]
        print(matrix)


# %%
utils.export_to_excel(matrix)