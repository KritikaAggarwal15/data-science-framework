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


# %%
sheet_name="Holiday_Package.csv"
target_variable='Holliday_Package'
modal=[['Decision_tree',10],['random_forest',5],['neural_network',10],
['logistic_regression',3],['lda',10],['knn',5],['naive_bayes',3,2,'f1'],['bagging',3,2,'f1'],
['ada_boost',3,2,'f1'],['gradient_boosting',3,2,'f1'],['support_vector_machine',3]]
modals_data = pd.DataFrame(modal,columns=['Modal','cross_validation','n_jobs','scoring'])


# %%
print(modals_data)

# %%
df_holiday_package=utils.read_file(sheet_name)
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

for i in modals_data.index:
    best_grid,X_train,X_test=modals.build_modal(modals_data.iloc[3],X_train,Y_train,X_test)
# %%
    Y_train_predict,Y_test_predict=modals.get_predicted_target_variable(best_grid,X_train,X_test)

# %%
    Y_train_predict_prob,Y_test_predict_prob=modals.get_predicted_target_variable_prob(best_grid,X_train,X_test)

# %%
    lr_train_auc=modals.get_auc_roc(best_grid,X_train,Y_train)
    print(lr_train_auc)


# %%
    lr_test_auc=modals.get_auc_roc(best_grid,X_test,Y_test)

# %%
    lr_train_accuracy=modals.get_accuracy(best_grid,X_train,Y_train)
    print(lr_train_accuracy)

# %%
    lr_test_accuracy=modals.get_accuracy(best_grid,X_test,Y_test)
    print(lr_test_accuracy)
# %%
    lr_train_precsion,lr_train_recall,lr_train_f1=modals.get_classification_report(Y_train,Y_train_predict)
# %%

    modals.get_confusion_matrix(Y_train,Y_train_predict)
# %%
    lr_test_precsion,lr_test_recall,lr_test_f1=modals.get_classification_report(Y_test,Y_test_predict)

# %%
    modals.get_confusion_matrix(Y_test,Y_test_predict)

# %%
    
# %%
    matrix["LR Train"]=[lr_train_accuracy,lr_train_auc,lr_train_recall,lr_train_precsion,lr_train_f1]
    matrix["LR Test"]=[lr_test_accuracy,lr_test_auc,lr_test_recall,lr_test_precsion,lr_test_f1]
    print(matrix)

# %%
