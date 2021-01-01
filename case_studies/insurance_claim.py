import pandas as pd
import numpy as np
import sys
import os
import traceback
from IPython.display import display
from utils import utils
from modals import modals
from bivariate_analysis import bivariate_analysis

sheet_name="insurance_claim.csv"
target_variable='Claimed'
modal=[['Decision_tree',10],['Random_forest',5,-1],['Neural_Network',10,-1],
['Logistic_Regression',3,-1],['LDA',10,-1],['KNN',5,-1],['Naive_Bayes',3,-1,'f1'],['Bagging',3,-1,'f1'],
['Ada_Boost',3,-1,'f1'],['Gradient_Boosting',3,-1,'f1'],['Support_Vector_Machine',3,-1]]
modals_data = pd.DataFrame(modal,columns=['Modal','cross_validation','n_jobs','scoring'])
display(modals_data)
display(modals_data.iloc[0])
df_insuarnce_claim = utils.read_file(sheet_name)
display(df_insuarnce_claim)

#df_insuarnce_claim=utils.drop_columns_by_colnums(df_insuarnce_claim,[0])
#display(df_insuarnce_claim)
utils.check_duplicate_data(df_insuarnce_claim)
categorical_columns=utils.get_categorical_cols(df_insuarnce_claim)
continous_columns=utils.get_continous_cols(df_insuarnce_claim)
utils.convert_cat_into_code(df_insuarnce_claim,categorical_columns)

modals.check_imbalance(df_insuarnce_claim[target_variable])

X,Y=modals.extract_target_column(df_insuarnce_claim,target_variable)

X_train, X_test, Y_train, Y_test = modals.split_data_into_train_test(X,Y,0.30,1)

print(Y_train.head())
print(Y_test.head())

modals.check_imbalance(Y_train)
modals.check_imbalance(Y_test)

modals.check_dimension_train_test(X_train, X_test, Y_train, Y_test)

best_grid,X_train,X_test=modals.build_modal(modals_data.iloc[10],X_train,Y_train,X_test)
print(X_train)
print(X_test)

y_train_predict,y_test_predict =modals.get_predicted_target_variable(best_grid,X_train,X_test)

y_train_predict_prob,y_test_predict_prob = modals.get_predicted_target_variable_prob(best_grid,X_train,X_test)

cart_train_auc = modals.get_auc_roc(best_grid,X_train,Y_train)

cart_test_auc = modals.get_auc_roc(best_grid,X_test,Y_test)

modals.get_confusion_matrix(Y_train,y_train_predict)
modals.get_confusion_matrix(Y_test,y_test_predict)

cart_train_accuracy = modals.get_accuracy(best_grid,X_train,Y_train)
cart_test_accuracy = modals.get_accuracy(best_grid,X_test,Y_test)
cart_train_precsion,cart_train_recall,cart_train_f1 = modals.get_classification_report(Y_train,y_train_predict)
cart_test_precsion,cart_test_recall,cart_test_f1 = modals.get_classification_report(Y_test,y_test_predict)

matrix=modals.create_comparision_matrix()
matrix["CART Train"]=[cart_train_accuracy,cart_train_auc,cart_train_recall,cart_train_precsion,cart_train_f1]
matrix["CART Test"]=[cart_test_accuracy,cart_test_auc,cart_test_recall,cart_test_precsion,cart_test_f1]
print(matrix)
#modals.get_feature_important(best_grid,X_train)

