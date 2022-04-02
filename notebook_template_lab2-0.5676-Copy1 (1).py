#!/usr/bin/env python
# coding: utf-8

# ## Load modules
# 
# In this cell, you can put all modules you use. You can use it to provide a clear code.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Your functions
# 
# In order to have a clear code, you can put all your own functions in this cell.

# In[2]:


def export_ens(df_test, pred_test, save=True, path_save="y_test_prediction.csv"):
    """
    Export submissions with the good ENS data challenge format.
    df_test : (pandas dataframe) test set
    proba_test : (numpy ndarray) prediction as a numpy ndarray you get using method .predict()
    save : (bool) if set to True, it will save csv submission in path_save path.
    path_save : (str) path where to save submission.
    return : dataframe for submission
    """
    df_submit = pd.Series(pred_test[:,0], index=df_test.index, name="spread")
    df_submit.to_csv(path_save, index=True)
    return df_submit

def check_test(result, expected, display_error):
    """
    Testing your results.
    """
    if result == expected:
        print("1 test passed.")
    else:
        print(display_error)


# ## Load datasets

# In[3]:


df_train = pd.read_csv(r'C:\Users\lizao\Downloads\lab2\data/input_training_imet9ZU.csv')
y_train = pd.read_csv(r'C:\Users\lizao\Downloads\lab2\data/output_training_yCN1f2d.csv')
df_test = pd.read_csv(r'C:\Users\lizao\Downloads\lab2\data/input_test_4AhEauI.csv')
df_full = pd.merge(df_train, y_train)
n_rows_train = df_train.shape[0]
n_rows_test = df_test.shape[0]


# In[4]:


check_test(n_rows_train, 629611, "wrong number of rows")
check_test(n_rows_test, 230304, "wrong number of rows")


# In[5]:


df_full['maturity'] = (df_full['dt_expiry']- df_full['dt_close'])
df_train['maturity'] = (df_train['dt_expiry']- df_train['dt_close'])
df_test['maturity'] = (df_test['dt_expiry']- df_test['dt_close'])


# In[6]:


df_full['hml'] = (np.log(df_full['high'])- np.log(df_full['open']))*(np.log(df_full['high'])- np.log(df_full['close'])) - (np.log(df_full['low'])- np.log(df_full['open']))*(np.log(df_full['low'])- np.log(df_full['close']))*100


# In[7]:


df_train['hml'] = (np.log(df_train['high'])- np.log(df_train['open']))*(np.log(df_train['high'])- np.log(df_train['close'])) - (np.log(df_train['low'])- np.log(df_train['open']))*(np.log(df_train['low'])- np.log(df_train['close']))*100
df_test['hml'] = (np.log(df_test['high'])- np.log(df_test['open']))*(np.log(df_test['high'])- np.log(df_test['close'])) - (np.log(df_test['low'])- np.log(df_test['open']))*(np.log(df_test['low'])- np.log(df_test['close']))*100


# ## Take a look on first rows
# 
# Take a look on the **df_train** first 5 rows using method .head().

# In[8]:


df_train.head(5)


# In[9]:


df_train = df_train.set_index(['ID'])
y_train = y_train.set_index('ID')
df_test = df_test.set_index('ID')


# In[10]:


mean_target_per_product = df_full.groupby(by = ["product_id", "liquidity_rank"])['spread'].mean()
mean_target_per_product.name = "mean_target_per_product"

df_train = df_train.merge(mean_target_per_product, how="left", right_index=True, left_on=["product_id", "liquidity_rank"])
df_test = df_test.merge(mean_target_per_product, how="left", right_index=True, left_on=["product_id", "liquidity_rank"])


# In[11]:



std_target_per_liquidity = df_full.groupby(by = [ "liquidity_rank"])['spread'].std()
std_target_per_liquidity.name = "std_target_per_liquidity"

df_train = df_train.merge(std_target_per_liquidity, how="left", right_index=True, left_on=["liquidity_rank"])
df_test = df_test.merge(std_target_per_liquidity, how="left", right_index=True, left_on=["liquidity_rank"])


# In[13]:


mean_hml = df_full.groupby('product_id')['hml'].mean()
mean_hml.name = 'mean_hml'

df_train = df_train.merge(mean_hml, how="left", right_index=True, left_on=["product_id"])
df_test = df_test.merge(mean_hml, how="left", right_index=True, left_on=["product_id"])


# In[14]:


#features_test = pd.DataFrame(df_test)
features_test = pd.DataFrame(df_test, columns= ['liquidity_rank',"volume", "fixed", "product_id", "normal_trading_day", "open_interest", "mean_target_per_product", "std_target_per_liquidity",  'maturity', 'tick_size', 'mean_hml', 'high', 'low', 'dt_close'])
#features_test = pd.DataFrame(df_test, columns= ["volume",  "hml", "fixed", "product_id", "normal_trading_day", "open_interest", "mean_target_per_product", "std_target_per_liquidity",  'maturity', 'tick_size', 'mean_hml'])


# In[15]:


#features_train = pd.DataFrame(df_train)
features_train = pd.DataFrame(df_train, columns= ['liquidity_rank', "volume",  "fixed", "product_id", "normal_trading_day", "open_interest", "mean_target_per_product", "std_target_per_liquidity",  'maturity', 'tick_size', 'mean_hml', 'high', 'low', 'dt_close'])
#features_train = pd.DataFrame(df_train, columns= ["volume",  "hml", "fixed", "product_id", "normal_trading_day", "open_interest", "mean_target_per_product", "std_target_per_liquidity",  'maturity', 'tick_size', 'mean_hml'])


# In[16]:


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(features_train, df_full['spread'])
x_test = features_test


# In[17]:


from sklearn.metrics import mean_squared_error
from math import sqrt

def compute_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


# In[18]:


from sklearn.ensemble import RandomForestRegressor
import math

# Prerequisites :
depths = range(24, 31)
estimators = range(200,420,20)
all_rmse_train = []
all_rmse_val = []
experiments  = []
all_rmse_train_tree = []
all_rmse_train_forest = []
all_rmse_val_tree = []
all_rmse_val_forest = []


for depth in depths: 
    for estimator in estimators:
        # Random Forest
        clf_forest = RandomForestRegressor(n_estimators=estimator, max_depth=depth, n_jobs=-1)
        model_forest = clf_forest.fit(x_train, y_train)

        pred_train = pd.Series(model_forest.predict(x_train), index=y_train.index)
        pred_val = pd.Series(model_forest.predict(x_val), index=y_val.index)


        # Compute MSLE evaluation metrics
        rmse_train = compute_rmse(y_train, pred_train)
        rmse_val = compute_rmse(y_val, pred_val)

        print("depth = %s |estimator = %s | RMSE train = %s | RMSE val = %s" % (depth, estimator, rmse_train, rmse_val))


# In[ ]:





# In[ ]:




