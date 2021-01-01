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
from bivariate_analysis import bivariate_analysis


# %%
df_gems = utils.read_file("cubic_zirconia.csv")
display(df_gems)


# %%
df_gems=utils.drop_columns_by_colnums(df_gems,[0])
display(df_gems)


# %%
categorical_cols=utils.get_categorical_cols(df_gems)
display(categorical_cols)


# %%
continous_cols=utils.get_continous_cols(df_gems)
display(continous_cols)


# %%
utils.describe(df_gems,continous_cols)


# %%
utils.describe(df_gems,categorical_cols)


# %%
utils.info(df_gems)


# %%
utils.shape(df_gems)


# %%
utils.check_null_values(df_gems)


# %%
utils.replace_null_values_with_mean(df_gems)
display(df_gems)


# %%
utils.check_null_values(df_gems)

# %%
utils.replace_null_values_with_mode(df_gems)
display(df_gems.head())


# %%
duplicates=(utils.check_duplicate_data(df_gems))
display(duplicates)

# %%
utils.drop_duplicate_data(df_gems)


# %%
utils.check_unique_value_categorical_var(df_gems[categorical_cols])



# %%
junk_data=utils.check_junk_value_cont_var(df_gems[continous_cols])
print(junk_data)


# %%
#utils.create_boxplot(df_gems[continous_cols])
# %%
##utils.univariate_analysis(df_gems,continous_cols)
# %%

bivariate_analysis.corr_coef(df_gems[continous_cols])

# %%

#bivariate_analysis.plot_corr_coef_heatMap(df_gems[continous_cols])

# %%
#bivariate_analysis.plot_corr_coef_pairPlot(df_gems[continous_cols])

# %%

#utils.remove_outliers(df_gems,continous_cols)
#utils.create_boxplot(df_gems[continous_cols])

utils.convert_cat_into_code(df_gems,categorical_cols)
display(df_gems)
print("cat",categorical_cols)
df_gems.info()

utils.check_unique_value_categorical_var(df_gems[categorical_cols])