import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import json
import math
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,roc_curve,classification_report,confusion_matrix,plot_confusion_matrix
from sklearn import metrics

class modals:

    def check_imbalance(self,dataframe):
        print(dataframe.value_counts(normalize=True))
        return

    def extract_target_column(self,dataframe, target):
        X = dataframe.drop([target], axis=1)
        Y = dataframe.pop(target)
        return X, Y

    def split_data_into_train_test(self,X, Y, test_size, random_state):
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state)
        return X_train, X_test, Y_train, Y_test

    def check_dimension_train_test(self,X_train, X_test, Y_train, Y_test):
        print('X_train', X_train.shape)
        print('X_test', X_test.shape)
        print('Y_train', Y_train.shape)
        print('Y_test', Y_test.shape)
        return

    def build_modal(self,modal_dataframe, X_train, Y_train, X_test):
        display("Modal Evaluation Started........")
        modal_name=modal_dataframe[1]
        cross_validation=modal_dataframe[2]
        if (modal_dataframe[3] is not None):
            n_jobs=int(modal_dataframe[3])
        else:
            n_jobs=modal_dataframe[3]

        scoring=modal_dataframe[4]
        sc = StandardScaler()
        min_max=MinMaxScaler()
        display("Model:",modal_name)
        os.chdir('C:\\Users\KRITIKA AGARWAL\\Documents\\Data Science\\Framework\\param_grid_data')
        if modal_name.lower() == "decision_tree":
            modal = DecisionTreeClassifier()
            f = open('decision_tree.json',)
            param_grid = json.load(f)

        elif modal_name.lower() == "random_forest":
            modal = RandomForestClassifier()
            f = open('random_forest.json',)
            param_grid = json.load(f)

        elif modal_name.lower() == "neural_network":
            modal = MLPClassifier()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            f = open('neural_network.json',)
            param_grid = json.load(f)
        
        elif modal_name.lower() == "logistic_regression":
            modal = LogisticRegression()
            f = open('logistic_regression.json',)
            param_grid = json.load(f)

        elif modal_name.lower() == "lda":
            modal = LinearDiscriminantAnalysis()
            f = open('lda.json',)
            param_grid = json.load(f)
            print(param_grid)

        elif modal_name.lower() == "knn":
            modal = KNeighborsClassifier()
            X_train = min_max.fit_transform(X_train)
            X_test = min_max.transform(X_test)
            f = open('knn.json',)
            param_grid = json.load(f)

        elif modal_name.lower() == "naive_bayes":
            modal = GaussianNB()
            f = open('naive_bayes.json',)
            param_grid = json.load(f)

        elif modal_name.lower() == "bagging":
            modal = BaggingClassifier()
            base_estimator={'base_estimator':[SVC(),DecisionTreeClassifier(),LogisticRegression()]}
            f = open('bagging.json',)
            param_grid = json.load(f)
            param_grid.update(base_estimator)
            print(param_grid)

        elif modal_name.lower() == "ada_boost":
            modal = AdaBoostClassifier()
            f = open('ada_boost.json',)
            param_grid = json.load(f)

        elif modal_name.lower() == "gradient_boosting":
            modal = GradientBoostingClassifier()
            f = open('gradient_boosting.json',)
            param_grid = json.load(f)

        elif modal_name.lower() == "support_vector_machine":
            modal = SVC(probability=True)
            X_train = pd.DataFrame(sc.fit_transform(X_train),columns=X_train.columns)
            X_test = pd.DataFrame(sc.transform(X_test),columns=X_test.columns)
            f = open('support_vector_machine.json',)
            param_grid = json.load(f)
        
        grid_search = GridSearchCV(estimator=modal, param_grid=param_grid, cv=cross_validation , n_jobs=n_jobs, scoring=scoring)
        grid_search.fit(X_train,Y_train)
        print("Best Param : ",grid_search.best_params_)
        best_grid = grid_search.best_estimator_
        return best_grid,X_train,X_test
   

    def get_predicted_target_variable(self,best_grid, X_train, X_test):
        y_train_predict = best_grid.predict(X_train)
        y_test_predict = best_grid.predict(X_test)
        display(pd.DataFrame(y_train_predict).head())
        display(pd.DataFrame(y_test_predict).head())
        return y_train_predict, y_test_predict

    def get_predicted_target_variable_prob(self,best_grid, X_train, X_test):
        y_train_predict_prob = best_grid.predict_proba(X_train)
        y_test_predict_prob = best_grid.predict_proba(X_test)
        display(pd.DataFrame(y_train_predict_prob).head())
        display(pd.DataFrame(y_test_predict_prob).head())
        return y_train_predict_prob, y_test_predict_prob

    def get_auc_roc(self,best_grid,X,Y):
        # predict probabilities
        probs = best_grid.predict_proba(X)
        # keep probabilities for the positive outcome only
        probs = probs[:, 1]
        # calculate AUC
        auc = round(roc_auc_score(Y,probs),2)
        print('Area Under Curve : %.3f' % auc)
        # calculate roc curve
        fpr, tpr, thresholds = roc_curve(Y,probs)
        plt.plot([0, 1], [0, 1], linestyle='--')
        # plot the roc curve for the model
        plt.plot(fpr, tpr)
        plt.title('Area Under Curve : %.3f' % auc)
        plt.show()
        return auc

    def get_confusion_matrix(self,Y, Y_predict):
        conf_matrix = confusion_matrix(Y, Y_predict)
        sns.heatmap(conf_matrix, annot=True, fmt='d',
                    cbar=False, cmap='rainbow')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.title('Confusion Matrix')
        plt.show()
        return

    def get_accuracy(self,best_grid, X, Y):
        accuracy =round(best_grid.score(X, Y),2)
        print('Accuracy : %.2f' % accuracy)
        return accuracy

    def get_classification_report(self,Y, Y_predict):
        print(classification_report(Y, Y_predict))
        metrics = classification_report(Y, Y_predict, output_dict=True)
        df = pd.DataFrame(metrics).transpose()
        precision = round(df.loc["1"][0],2)
        recall = round(df.loc["1"][1],2)
        f1_score = round(df.loc["1"][2],2)
        print('precision ', precision)
        print('recall ', recall)
        print('f1 score ', f1_score)
        return precision, recall, f1_score

    def print_modal_conclusion(self,matrix):
        print("Train Data")
        print("AUC:",round(matrix.loc[0][0],3)*100, "%")
        print("Accuracy:", round(matrix.loc[1][0],3)*100, "%")
        print("Sensitivity:", round(matrix.loc[2][0],3)*100, "%")
        print("Precision:", round(matrix.loc[3][0],3)*100, "%")
        print("f1-Score:", round(matrix.loc[4][0],3)*100, "%")
        # print("\n")
        # print("Test Data")
        # print("AUC:", round(test_auc, 3)*100, "%")
        # print("Accuracy:", round(test_acc, 3)*100, "%")
        # print("Sensitivity:", round(recall, 3)*100, "%")
        # print("Precision:", round(precision, 3)*100, "%")
        # print("f1-Score:", round(test_f1, 3)*100, "%")
        return

    def create_comparision_matrix(self):
        index=['Accuracy', 'AUC', 'Recall','Precision','F1 Score']
        data = pd.DataFrame(index=index)
        return data

    def get_feature_important(self,best_grid,X_train):
        print (pd.DataFrame(best_grid.feature_importances_,columns = ["Imp"], index = X_train.columns).sort_values('Imp',ascending=False))
        return