#!/usr/bin/env python
# coding: utf-8

# ### By Becky Tsang
# ### Date: March 21, 2026
# ### Project - Global coffee analysis based on location, spending habits, payment method, weather conditions and transaction amounts.

# ## Libraries

# In[37]:


#Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


# ### Read the data file containing coffee shop sales

# In[38]:


df = pd.read_csv('coffee_shop_sales.csv')


# In[59]:


df.head()


# In[40]:


# List the columns from the table
df.columns


# In[41]:


# Information of the columns
df.info()


# # Data Cleaning

# In[42]:


# Check for null values
df.isnull().sum()


# In[43]:


# Drop unwanted column - holiday_name
df = df.drop(columns=['holiday_name'])


# In[44]:


# Drop rows with null values
df_clean = df.dropna(subset=['customer_age_group', 'customer_gender', 'weather_condition', 'temperature_c'])


# In[45]:


#Check if there are any null values
print(df_clean.isnull().sum())


# In[46]:


# Check for duplicate entries
df_clean.duplicated().sum()


# ## Summary

# In[47]:


df_clean.describe()


# ## Exploratory Data Analysis

# In[85]:


fig = plt.figure(figsize=(10,6))
sns.countplot(data=df_clean, x='payment_method')

plt.title('Usage of Different Payment Methods')
plt.xlabel('Payment Method')
plt.ylabel('Count')

plt.show()


# ### Credit card is the preferrred payment method. Mobile wallet is the least used. 

# In[86]:


plt.figure(figsize=(10,6))

sns.countplot(data=df_clean, x='product_category')

plt.title('Number of Transactions by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Count')

plt.show()


# ### Based on the above bar graph, coffee was the most purchased.

# In[87]:


sns.scatterplot(data=df, x='unit_price', y='total_amount', alpha=0.5)


# ### Higher-priced items often lead to higher total spending

# In[88]:


sns.boxplot(data=df, x='product_category', y='total_amount')
plt.title('Transaction Amount by Product Category')
plt.xticks(rotation=30)
plt.show()


# ### Merchandise has the highest transaction amounts overall. Spending is not normally distributed. There are many outliers with most transaction amounts positively skewed.

# In[89]:


df_filtered = df[df['total_amount'] > 90]
df_filtered


# ### Two female customers made purchases above $90.00. 

# In[90]:


plt.figure(figsize=(10,6))
sns.countplot(data=df, x='weather_condition')

plt.title('Distribution of Transactions Across Weather Conditions')
plt.xlabel('Weather Condition')
plt.ylabel('Count')

plt.show()


# ### More transactions are made on rainy and sunny days. Customers are less active when it snows.

# In[91]:


plt.figure(figsize=(10,6))
sns.countplot(data=df, x='country')

plt.title('Distribution of Transactions by country')
plt.xlabel('county')
plt.ylabel('Count')

plt.show()


# ### United State consumers makes up a significant amount of the sales transactions.

# ### Key insights: Based on the Global Coffee dataset, United State makes the most transactions. Bigger sales are made on merchandise due to higher unit price however most transactions come from coffee consumptions. Credit card is the preferred payment method, and there appears to be more customers on rainy and sunny days.
