Apprentice Chef Case Study - Machine Learning | Python

**Introduction**

The two main goals of this following Report are to investigate factors that affect Apprentice chef's revenue and analyze what characteristics of customers that likely to subscribe to the Half Way There Campaign subscription.

**Part 1: Business Report**

**1  Predicting Revenue**

- Data Preparation
```
#############################################
# import necessary libraries
#############################################
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import seaborn           as sns
import random            as rand                    
import scipy             as sp
import sklearn.linear_model
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
# CART model packages
from sklearn.tree import DecisionTreeClassifier     
from sklearn.tree import export_graphviz             
from six import StringIO                             
from IPython.display import Image                    
import pydotplus 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score  

#############################################
# import and prepare dataset
#############################################
# import, adjust and explore Apprentice_Chef Dataset 
file = 'Apprentice_Chef_Dataset.xlsx'
Apprentice_Chef = pd.read_excel(file)

# create copy version of original data
dataset = Apprentice_Chef.copy()
dataset.columns = dataset.columns.str.lower()

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

```
- Featur Engineering
```
#############################################
# Feature Engineering
#############################################
# apply log transformation  
# continue Variable
dataset['log_revenue'] = np.log10(dataset['revenue'])
dataset['log_avg_prep_vid_time'] = np.log10(dataset['avg_prep_vid_time'])
dataset['log_avg_time_per_site_visit'] = np.log10(dataset['avg_time_per_site_visit'])

# count variables
dataset['log_total_photos_viewed'] = np.log10(dataset['total_photos_viewed']+0.01)
dataset['log_total_meals_ordered'] = np.log10(dataset['total_meals_ordered'])

# new variables
dataset['complain_rate'] = dataset['contacts_w_customer_service'] / dataset['total_meals_ordered']
dataset['log_complain_rate'] = np.log(dataset['complain_rate']+0.0001)

# create new columns for dummy variable
dataset['has_ordered_total_meals_ordered']   = 0
dataset['single_unique_meals_purch']   = 0

# for loop to subsetting the threshold for each variable
for index, value in dataset.iterrows():
    # total_meals_ordered
    if dataset.loc[index,'log_total_meals_ordered'] > 1.25:
        dataset.loc[index, 'has_ordered_total_meals_ordered'] = 1
    # unique_meals_purch
    if dataset.loc[index,'unique_meals_purch'] == 1:
        dataset.loc[index, 'single_unique_meals_purch'] = 1  

#############################################
# Preparing set of x variables and y variable 
#############################################       
x_var = ['cross_sell_success', 
        'median_meal_rating',
        'contacts_w_customer_service',
        'largest_order_size',
        'log_avg_prep_vid_time', 
        'log_avg_time_per_site_visit', 
        'log_complain_rate', 
        'log_total_photos_viewed',
        'has_ordered_total_meals_ordered',
        'single_unique_meals_purch']

# preparing explanatory variable data
ols_data = dataset[x_var]

# preparing response variables
log_dataset_target = dataset.loc[ : , 'log_revenue']

#############################################
# OLS model
#############################################    
# split train and test data on OLS p-value x-dataset
x_train_ols, x_test_ols, y_train_ols, y_test_ols = train_test_split(
            ols_data,
            log_dataset_target,
            test_size = 0.25,
            random_state = 219)

# instantiating a model object
lr = LinearRegression()

# fit the model to the training data
lr_fit = lr.fit(x_train_ols, y_train_ols)

# predicting on new data
lr_pred = lr_fit.predict(x_test_ols)

# storing training Testing score, gap
lr_train_score = lr.score(x_train_ols, y_train_ols).round(3)
lr_test_score  = lr.score(x_test_ols, y_test_ols).round(3)
lr_test_gap = abs(lr_train_score - lr_test_score).round(3)

# storing model coefficients 
lr_model_values = zip(dataset[x_var].columns,
                  lr_fit.coef_.round(decimals = 2))
lr_model_lst = [('intercept', lr_fit.intercept_.round(decimals = 2))]

# printing out each feature-coefficient pair one by one
for val in lr_model_values:
    lr_model_lst.append(val)
   
```
1.1 OLS Regression Model Output
```
OLS Regression Model Output
------------------------------------------------------------------------------  
Model Type  Training  Testing  Train-Test Gap  Model Size
       OLS     0.799    0.801           0.002          11
------------------------------------------------------------------------------ 
                       Variables  Coefficients
                       intercept          1.19
           log_avg_prep_vid_time          0.58
       single_unique_meals_purch          0.26
 has_ordered_total_meals_ordered          0.21
              median_meal_rating          0.06
     log_avg_time_per_site_visit          0.06
     contacts_w_customer_service          0.03
         log_total_photos_viewed          0.01
              cross_sell_success         -0.01
              largest_order_size         -0.01
               log_complain_rate         -0.08
------------------------------------------------------------------------------
The strongest impact feature
------------------------------------------------------------------------------
             Variables  Coefficients
 log_avg_prep_vid_time          0.58
 
```
**1.2 Interpreting Result**

With the regression analysis, 80.1% (R-Square) of the total variance is explained by the model. Average time in seconds of meal preparation video has the strongest impact on revenue as well as highest correlation (shown in Table 1). We could say that every additional one-percent increase in the average time of the video is associated with about a 0.58 percent change in revenue.

**2. Predicting Cross Sell Success with Classification Tree Model**

- Feature Engineering
```
#############################################
# Feature Engineering
# engineering 'email' variable
#############################################
# create empty list for storing email domain
placeholder_lst = []

# looping over each email address
for index, col in dataset.iterrows():
    
    split_email = dataset.loc[index, 'email'].split(sep = '@')
    placeholder_lst.append(split_email)
    
email_df = pd.DataFrame(placeholder_lst)


# renaming column to concatenate ('1' => 'personal_email_domain')
email_df.columns = ['0' , 'email_domain']

# concatenating personal_email_domain with dataset DataFrame
dataset = pd.concat([dataset, email_df['email_domain']],
                     axis = 1)
# email domain types
per_domain = ['@gmail.com', '@yahoo.com', '@protonmail.com']

pro_domain = ['@mmm.com', '@amex.com', '@apple.com', '@boeing.com', 
              '@caterpillar.com', '@chevron.com', '@cisco.com', 
              '@cocacola.com', '@disney.com', '@dupont.com', 
              '@exxon.com', '@ge.org', '@goldmansacs.com', 
              '@homedepot.com', '@ibm.com', '@intel.com', 
              '@jnj.com', '@jpmorgan.com', '@mcdonalds.com', 
              '@merck.com', '@microsoft.com', '@nike.com', 
              '@pfizer.com', '@pg.com', '@travelers.com', 
              '@unitedtech.com', '@unitedhealth.com', '@verizon.com', 
              '@visa.com', '@walmart.com']

junk_domain = ['@me.com', '@aol.com', '@hotmail.com', 
               '@live.com', '@msn.com', '@passport.com']

# create a placeholder list to store email domain types
placeholder_lst = []

# looping to group observations by domain type
for domain in dataset['email_domain']:
        if '@' + domain in per_domain :
            placeholder_lst.append(1)
        
        elif '@' + domain in pro_domain :
            placeholder_lst.append(1)
            
        else:
            placeholder_lst.append(0)


# concatenating with original DataFrame
dataset['valid_email'] = pd.Series(placeholder_lst)

#############################################
# Feature Engineering
# engineering 'name' variable
#############################################

def text_split_feature(col, df, sep=' ', new_col_name='number_of_names'):
    """
Splits values in a string Series (as part of a DataFrame) and sums the number
of resulting items. Automatically appends summed column to original DataFrame.

PARAMETERS
----------
col          : column to split
df           : DataFrame where column is located
sep          : string sequence to split by, default ' '
new_col_name : name of new column after summing split, default
               'number_of_names'
"""
    
    df[new_col_name] = 0
    
    
    for index, val in df.iterrows():
        df.loc[index, new_col_name] = len(df.loc[index, col].split(sep = ' '))

# calling text_split_feature
text_split_feature(col = 'name',
                   df  = dataset)

#############################################
# create dummies variable 
#############################################
# avg_prep_vid_time
thres_avg_prep_vid_time = 253
dataset['lessthan253_avg_prep_vid_time'] = 0

# cancellations_before_noon
thres_cancellations_before_noon = 1.5
dataset['lessthan2_cancellations_before_noon'] = 0

# master_classes_attended
thres_master_classes_attended = 0
dataset['has_attended_master_classes']   = 0

for index, value in dataset.iterrows():
    # avg_prep_vid_time
    if dataset.loc[index,'avg_prep_vid_time'] < thres_avg_prep_vid_time:
        dataset.loc[index, 'lessthan253_avg_prep_vid_time'] = 1
        
    # avg_clicks_per_visit
    if dataset.loc[index,'cancellations_before_noon'] < thres_cancellations_before_noon:
        dataset.loc[index, 'lessthan2_cancellations_before_noon'] = 1
    
    # master_classes_attended
    if dataset.loc[index,'master_classes_attended'] > thres_master_classes_attended :
        dataset.loc[index, 'has_attended_master_classes'] = 1
        
        
        
# create new columns for dummy variable
dataset['mobile_phone_user'] = 0

# for loop to subsetting the threshold for this variable
for index, value in dataset.iterrows():
    if dataset.loc[index,'mobile_number'] == 1 and\
       dataset.loc[index,'mobile_logins'] > 0 :
        dataset.loc[index,'mobile_phone_user'] = 1
        
        
        
# create new columns for dummy variable
dataset['have_packagelocker_fridgelocker'] = 0

# for loop to subsetting the threshold for this variable
for index, value in dataset.iterrows():
    if dataset.loc[index,'package_locker'] == 1 and dataset.loc[index,'refrigerated_locker'] == 1 :
        dataset.loc[index, 'have_packagelocker_fridgelocker'] = 1

########################################
# explanatory variable sets
########################################
candidate_dict = {
   'x_var' : ['valid_email',
              'tastes_and_preferences',
              'has_attended_master_classes',  
              'number_of_names', 
              'mobile_phone_user',
              'lessthan2_cancellations_before_noon', 
              'lessthan253_avg_prep_vid_time',
              'have_packagelocker_fridgelocker']
}
########################################
# Classification Tree Model : Pruned Tree
########################################

# train/test split with the full model
dataset_data   =  dataset.loc[ : , candidate_dict['x_var']]
dataset_target =  dataset.loc[ : , 'cross_sell_success']


# This is the exact code we were using before
x_train, x_test, y_train, y_test = train_test_split(
            dataset_data,
            dataset_target,
            test_size    = 0.25,
            random_state = 219,
            stratify     = dataset_target)

########################################
# Classification Tree Model : Pruned Tree
########################################
# INSTANTIATING a classification tree object
pruned_tree = DecisionTreeClassifier(max_depth = 4,
                                     min_samples_leaf = 25,
                                     random_state = 219)

# FITTING the training data
pruned_tree_fit  = pruned_tree.fit(x_train, y_train)

# PREDICTING on new data
pruned_tree_pred = pruned_tree_fit.predict(x_test)

# saving scoring data for future use
pruned_tree_train_score = pruned_tree_fit.score(x_train, y_train).round(3) # accuracy
pruned_tree_test_score  = pruned_tree_fit.score(x_test, y_test).round(3)   # accuracy

# saving auc score
pruned_tree_auc_score   = roc_auc_score(y_true  = y_test,
                                        y_score = pruned_tree_pred).round(3) # auc

# unpacking the confusion matrix
pruned_tree_tn, \
pruned_tree_fp, \
pruned_tree_fn, \
pruned_tree_tp = confusion_matrix(y_true = y_test, y_pred = pruned_tree_pred).ravel()

########################################
# plot_feature_importances
########################################
def plot_feature_importances(model, train, export = False):
    """
    Plots the importance of features from a CART model.
    
    PARAMETERS
    ----------
    model  : CART model
    train  : explanatory variable training data
    export : whether or not to export as a .png image, default False
    """
    
    # declaring the number
    n_features = x_train.shape[1]
    
    # setting plot window
    fig, ax = plt.subplots(figsize=(12,9))
    
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')

```
**2.1 Pruned Tree Model Output**

```
Pruned Tree Model Output
------------------------------------------------------------------------------  
  Model Name  AUC Score  Training Accuracy  Testing Accuracy   Confusion Matrix
 Pruned Tree      0.736              0.748             0.786  (93, 63, 41, 290)
------------------------------------------------------------------------------ 
```
**2.2 Interpreting Result**

From the model, 73.6% (AUC Score) of the model prediction result was accurately classified customers who accepted and not accepted the campaign. The most important factor describing customers who likely to subscribe to the campaign is customers whose registered email are either personal or professional email domain. The second is a number of names specified by a count after a split of the space bar (Shown in Figure 1). However, the number of names needed more investigation whether the length of the name refers to nationality, randomness, or any possible factors.

**3  Conclusion**

Due to evidence from the revenue regression model analysis, it is clear that the duration of the meal preparation video has an impact on customer's buying decisions. The video is the medium that takes advantage of advance and engaging communication, therefore Apprentice chefs should put more focus on developing and creating more meal preparation videos with quality contents and longer duration.

To improve the acquisition for subscribers of the Half-Way-There Campaign, Apprentice chef should avoid targeting junk email and primarily target customers whose email domain is either personal or professional and customers whose full names are long.

**Appendix**

**Table 1**
```
Table 1
------------------------------------------------------------------------------  
The table below shown the correlation between explanatory variables 
and response variable.
------------------------------------------------------------------------------ 

                                 log_revenue
log_revenue                             1.00
log_avg_prep_vid_time                   0.67
median_meal_rating                      0.65
has_ordered_total_meals_ordered         0.52
largest_order_size                      0.45
log_total_photos_viewed                 0.41
single_unique_meals_purch               0.21
log_avg_time_per_site_visit             0.15
cross_sell_success                      0.01
contacts_w_customer_service            -0.04
log_complain_rate                      -0.57
```
**Figure 1**
```
Figure 1
------------------------------------------------------------------------------  
The graph below shown the importance level of each factor taking into account
with the model.The most important feature is 'Valid Email' following by
'Number Of Names'.
------------------------------------------------------------------------------ 
```

- Coding for printing Appendix
```
# Table 1 Correlation
###############################
# creating a (Pearson) correlation matrix
df_corr = dataset[['log_revenue','cross_sell_success', 
        'median_meal_rating',
        'contacts_w_customer_service',
        'largest_order_size',
        'log_avg_prep_vid_time', 
        'log_avg_time_per_site_visit', 
        'log_complain_rate', 
        'log_total_photos_viewed',
        'has_ordered_total_meals_ordered',
        'single_unique_meals_purch']].corr().round(2)


# printing (Pearson) correlations with SalePrice
corr = df_corr.loc['log_revenue'].sort_values(ascending = False)
corr = pd.DataFrame(corr)
print(f"""
Table 1
------------------------------------------------------------------------------  
The table below shown the correlation between explanatory variables 
and response variable.
------------------------------------------------------------------------------ 
""")
print(corr)

# Figure 1 Feature Importances
###############################
print(f"""
Figure 1
------------------------------------------------------------------------------  
The graph below shown the importance level of each factor taking into account
with the model.The most important feature is 'Valid Email' following by
'Number Of Names'.
------------------------------------------------------------------------------ 
""")
# plotting feature importance
plot_feature_importances(pruned_tree_fit,train  = x_train, export = False)

```




