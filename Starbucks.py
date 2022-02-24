#!/usr/bin/env python
# coding: utf-8

# ## Portfolio Exercise: Starbucks
# <br>
# 
# <img src="https://opj.ca/wp-content/uploads/2018/02/New-Starbucks-Logo-1200x969.jpg" width="200" height="200">
# <br>
# <br>
#  
# #### Background Information
# 
# The dataset you will be provided in this portfolio exercise was originally used as a take-home assignment provided by Starbucks for their job candidates. The data for this exercise consists of about 120,000 data points split in a 2:1 ratio among training and test files. In the experiment simulated by the data, an advertising promotion was tested to see if it would bring more customers to purchase a specific product priced at $10. Since it costs the company 0.15 to send out each promotion, it would be best to limit that promotion only to those that are most receptive to the promotion. Each data point includes one column indicating whether or not an individual was sent a promotion for the product, and one column indicating whether or not that individual eventually purchased that product. Each individual also has seven additional features associated with them, which are provided abstractly as V1-V7.
# 
# #### Optimization Strategy
# 
# Your task is to use the training data to understand what patterns in V1-V7 to indicate that a promotion should be provided to a user. Specifically, your goal is to maximize the following metrics:
# 
# * **Incremental Response Rate (IRR)** 
# 
# IRR depicts how many more customers purchased the product with the promotion, as compared to if they didn't receive the promotion. Mathematically, it's the ratio of the number of purchasers in the promotion group to the total number of customers in the purchasers group (_treatment_) minus the ratio of the number of purchasers in the non-promotional group to the total number of customers in the non-promotional group (_control_).
# 
# $$ IRR = \frac{purch_{treat}}{cust_{treat}} - \frac{purch_{ctrl}}{cust_{ctrl}} $$
# 
# 
# * **Net Incremental Revenue (NIR)**
# 
# NIR depicts how much is made (or lost) by sending out the promotion. Mathematically, this is 10 times the total number of purchasers that received the promotion minus 0.15 times the number of promotions sent out, minus 10 times the number of purchasers who were not given the promotion.
# 
# $$ NIR = (10\cdot purch_{treat} - 0.15 \cdot cust_{treat}) - 10 \cdot purch_{ctrl}$$
# 
# For a full description of what Starbucks provides to candidates see the [instructions available here](https://drive.google.com/open?id=18klca9Sef1Rs6q8DW4l7o349r8B70qXM).
# 
# Below you can find the training data provided.  Explore the data and different optimization strategies.
# 
# #### How To Test Your Strategy?
# 
# When you feel like you have an optimization strategy, complete the `promotion_strategy` function to pass to the `test_results` function.  
# From past data, we know there are four possible outomes:
# 
# Table of actual promotion vs. predicted promotion customers:  
# 
# <table>
# <tr><th></th><th colspan = '2'>Actual</th></tr>
# <tr><th>Predicted</th><th>Yes</th><th>No</th></tr>
# <tr><th>Yes</th><td>I</td><td>II</td></tr>
# <tr><th>No</th><td>III</td><td>IV</td></tr>
# </table>
# 
# The metrics are only being compared for the individuals we predict should obtain the promotion – that is, quadrants I and II.  Since the first set of individuals that receive the promotion (in the training set) receive it randomly, we can expect that quadrants I and II will have approximately equivalent participants.  
# 
# Comparing quadrant I to II then gives an idea of how well your promotion strategy will work in the future. 
# 
# Get started by reading in the data below.  See how each variable or combination of variables along with a promotion influences the chance of purchasing.  When you feel like you have a strategy for who should receive a promotion, test your strategy against the test dataset used in the final `test_results` function.

# In[175]:


# load in packages
from itertools import combinations

from test_results import test_results, score
import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk

import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')

# load in the data
train_data = pd.read_csv('./training.csv')
train_data.head()


# In[176]:


train_data.shape


# In[177]:


len(train_data['ID'].unique())


# In[178]:


train_data.describe()


# In[179]:


train_data['V1'].value_counts()


# In[180]:


# V1, V4, V5, V6 and V7 are categorical variables


# In[181]:


test_data = pd.read_csv('./Test.csv')
test_data.head()


# In[182]:


test_data.shape


# In[183]:


len(test_data['ID'].unique())


# In[184]:


# any missing data?
train_data.isnull().mean()


# In[185]:


# any missing data?
test_data.isnull().mean()


# In[186]:


# what is the purchase rate with/out promotion in train data?
pd.pivot_table(train_data, values='purchase', index = 'Promotion')


# In[187]:


# what is the purchase rate with/out promotion?
pd.pivot_table(test_data, values='purchase', index = 'Promotion')


# In[188]:


train2 = train_data[train_data['Promotion']=='Yes']
train2['sendpro']=[1 if x==1 else 0 for x in train2['purchase']]
train2['sendpro'].value_counts()


# In[189]:


# checking logic
print('marked send pro, and was right')
print(train2[(train2['Promotion']=='Yes') & (train2['purchase']==1)]['sendpro'].sum())
print('marked send pro, and was wrong')
print(train2[(train2['Promotion']=='Yes') & (train2['purchase']==0)]['sendpro'].sum())


# In[191]:


train3 = train2.drop(['Promotion','purchase','ID'],axis=1)
X_train = train3.drop('sendpro',axis=1)
y_train = train3['sendpro']
X_test = test_data.drop(['Promotion','purchase','ID'],axis=1)


# In[194]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred


# In[195]:


y_pred.sum()


# In[144]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred


# In[145]:


y_pred.sum()


# In[146]:


len(y_pred)


# In[147]:


test_data['sendpro']=y_pred
print('suggested send pro, and was right')
print(test_data[(test_data['Promotion']=='Yes') & (test_data['purchase']==1)]['sendpro'].sum())
print('suggested send pro, and was wrong')
print(test_data[(test_data['Promotion']=='Yes') & (test_data['purchase']==0)]['sendpro'].sum())


# In[148]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
y_pred2 = clf.predict(X_test)
y_pred2


# In[149]:


y_pred.sum()


# In[150]:


len(y_pred)


# In[151]:


test_data['sendpro']=y_pred2
print('suggested send pro, and was right')
print(test_data[(test_data['Promotion']=='Yes') & (test_data['purchase']==1)]['sendpro'].sum())
print('suggested send pro, and was wrong')
print(test_data[(test_data['Promotion']=='Yes') & (test_data['purchase']==0)]['sendpro'].sum())


# In[24]:


import pandas as pd
import numpy as np

def score(df, promo_pred_col = 'Promotion'):
    n_treat       = df.loc[df[promo_pred_col] == 'Yes',:].shape[0]
    n_control     = df.loc[df[promo_pred_col] == 'No',:].shape[0]
    n_treat_purch = df.loc[df[promo_pred_col] == 'Yes', 'purchase'].sum()
    n_ctrl_purch  = df.loc[df[promo_pred_col] == 'No', 'purchase'].sum()
    irr = n_treat_purch / n_treat - n_ctrl_purch / n_control
    nir = 10 * n_treat_purch - 0.15 * n_treat - 10 * n_ctrl_purch
    return (irr, nir)
    

def test_results(promotion_strategy):
    test_data = pd.read_csv('Test.csv')
    df = test_data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']]
    promos = promotion_strategy(df)
    score_df = test_data.iloc[np.where(promos == 'Yes')]    
    irr, nir = score(score_df)
    print("Nice job!  See how well your strategy worked on our test data below!")
    print()
    print('Your irr with this strategy is {:0.4f}.'.format(irr))
    print()
    print('Your nir with this strategy is {:0.2f}.'.format(nir))
    
    print("We came up with a model with an irr of {} and an nir of {} on the test set.\n\n How did you do?".format(0.0188, 189.45))
    return irr, nir


# In[ ]:





# In[ ]:





# In[118]:


def promotion_strategy(df):
    '''
    INPUT 
    df - a dataframe with *only* the columns V1 - V7 (same as train_data)

    OUTPUT
    promotion_df - np.array with the values
                   'Yes' or 'No' related to whether or not an 
                   individual should recieve a promotion 
                   should be the length of df.shape[0]
                
    Ex:
    INPUT: df
    
    V1	V2	  V3	V4	V5	V6	V7
    2	30	-1.1	1	1	3	2
    3	32	-0.6	2	3	2	2
    2	30	0.13	1	1	4	2
    
    OUTPUT: promotion
    
    array(['Yes', 'Yes', 'No'])
    indicating the first two users would recieve the promotion and 
    the last should not.
    '''    
    promotion = []
    
    lendf = df.shape[0]
    
    for i in range(lendf):
        if y_pred[i] == 1:
            promotion.append('Yes')
        else:
            promotion.append('No')
        
    promotion = np.array(promotion)
    
    return promotion


# In[ ]:





# In[119]:


# This will test your results, and provide you back some information 
# on how well your promotion_strategy will work in practice

test_results(promotion_strategy)


# In[ ]:




