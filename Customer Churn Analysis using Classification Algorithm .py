#!/usr/bin/env python
# coding: utf-8

# In[72]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
le=LabelEncoder()
from sklearn.metrics import r2_score, classification_report, mean_absolute_error
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


# In[4]:


Churn_data=pd.read_csv('customer churn.csv')


# In[5]:


Churn_data.head()


# In[6]:


Churn_data.shape


# In[7]:


Churn_data.info()


# # EDA

# In[8]:


Churn_data.size


# In[9]:


Churn_data.describe()


# In[10]:


Churn_data.columns


# In[11]:


Churn_data.isnull().sum()


# In[12]:


Churn_data.isna().sum()


# In[13]:


print(min(Churn_data.tenure))
print(Churn_data.tenure.mean())
print(max(Churn_data.tenure))


# In[14]:


print(min(Churn_data.MonthlyCharges))
print(Churn_data.MonthlyCharges.mean())
print(max(Churn_data.MonthlyCharges))


# In[15]:


for i in Churn_data.columns:
  if Churn_data[i].dtypes != object:
    plt.boxplot(Churn_data[i])
    plt.xlabel(i)
    plt.show()


# # DATA ANALYSIS AND MANIPULATION

# In[16]:


print(min(Churn_data.tenure))
print(Churn_data.tenure.mean())
print(max(Churn_data.tenure))


# In[17]:


print(min(Churn_data.MonthlyCharges))
print(Churn_data.MonthlyCharges.mean())
print(max(Churn_data.MonthlyCharges))


# In[18]:


senior_male_electronics = Churn_data[(Churn_data['gender'] == 'Male') & (Churn_data['SeniorCitizen'] == 1) &
                               (Churn_data['PaymentMethod'] == 'Electronic check')]

senior_male_electronics.count().unique().sum()


# In[19]:


#Extract all those customers whose tenure is greater than 70 months or their Monthly charges is more than 100$ & store the result in ‘customer_total_tenure’


# In[20]:


customer_total_tenure=Churn_data[(Churn_data['tenure']>70) | (Churn_data['MonthlyCharges']>100)]
customer_total_tenure.count().unique().sum()


# In[21]:


#Extract all the customers whose Contract is of two years, payment method is Mailed check & the value of Churn is ‘Yes’ 
#& store the result in ‘two_mail_yes’


# In[22]:


Two_mail_yes=Churn_data[(Churn_data['Contract']=='Two year') & (Churn_data['PaymentMethod']=='Mailed check') & (Churn_data['Churn']=='Yes')]
Two_mail_yes.count().unique().sum()


# In[23]:


#Store number 333 customer in variable Customer_333
Customer_333=Churn_data.sample(332)
Customer_333.sample(1)


# In[24]:


#Churn data Partition
Churn_data.Churn.value_counts()


# In[25]:


Churn_data.head(5)


# # Data Analysis and Data Visualization

# In[26]:


sns.barplot(data=Churn_data, y='tenure', x='Churn')


# In[27]:


sns.barplot(data=Churn_data, y='MonthlyCharges', x='Churn')


# In[28]:


sns.barplot(data=Churn_data, y='MonthlyCharges', x='InternetService')


# In[29]:


x=Churn_data.Contract.value_counts().index
y=Churn_data.Contract.value_counts()


# In[30]:


plt.bar(x,y, color=["Blue",'green', 'red', 'orange'])
plt.xlabel('Contract Categories')
plt.ylabel('Contract Count')
plt.show()
plt.tight_layout()


# In[31]:


plt.title('Distribution of tenure')
plt.hist(Churn_data.tenure, bins = 30, color = 'green')
plt.tight_layout();


# In[32]:


sns.scatterplot(x=Churn_data.tenure, y=Churn_data.MonthlyCharges, color='blue', s=100, alpha=0.6)
plt.show()
plt.tight_layout()


# In[33]:


sns.boxplot(x = 'Contract' , y = 'tenure', data = Churn_data)
plt.tight_layout();


# In[34]:


x1=Churn_data.PaymentMethod.value_counts().index
y1=Churn_data.PaymentMethod.value_counts()

plt.bar(x1,y1, color=["Blue",'green', 'red','purple'])
plt.xlabel('Contract Categories')
plt.ylabel('Contract Count')
plt.show()
plt.tight_layout()


# # Converting categorical variables to dummy using get_dummies

# In[65]:


X=Churn_data.drop(['Churn','TotalCharges', 'customerID', 'gender'], axis=1)
y=le.fit_transform(Churn_data['Churn'])


# In[66]:


df_dummies=pd.get_dummies(X.select_dtypes(include=['object']))
df_dummies


# In[67]:


X=pd.concat([X.drop(X.select_dtypes(include=['object']).columns, axis=1), df_dummies],axis=1)


# In[68]:


y.shape


# In[69]:


X


# In[73]:


# a. Recursive Feature Elimination (RFE) using Logistic Regression
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=5)  # Select the top 5 features
rfe.fit(X, y)
# Print the selected features by RFE
selected_rfe_features = X.columns[rfe.support_]
print("\nSelected Features by RFE:", selected_rfe_features)

# b. Feature Importance using RandomForest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)

# Get feature importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort in descending order

# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
plt.title('Feature Importance from RandomForest')
plt.bar(range(10), importances[indices][:10], align='center')
plt.xticks(range(10), X.columns[indices][:10], rotation=90)
plt.tight_layout()
plt.show()

# Print the top 10 most important features
print("\nTop 10 Most Important Features from RandomForest:\n", X.columns[indices][:10])


# In[76]:


y1


# In[79]:


y = Churn_data[['Churn']]
x = Churn_data[['MonthlyCharges', 'tenure']]


# In[80]:


x_test, x_train, y_test, y_train= train_test_split(x,y1, test_size=0.35,random_state=20)


# # Building Model
# 

# In[82]:


LM=LogisticRegression()


# In[ ]:


LM.fit(x_)

