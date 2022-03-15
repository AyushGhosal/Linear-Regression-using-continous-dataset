#!/usr/bin/env python
# coding: utf-8

# In[1]:


#LINEAR REGRESSION FOR CONTINUOS VALUE USING BOSTON DATASET
from sklearn.datasets import load_boston


# In[2]:


boston = load_boston()


# In[3]:


boston.DESCR


# In[4]:


import pandas as pd


# In[5]:


data=pd.DataFrame(boston.data, columns=boston.feature_names)
data


# In[6]:


data['MEDV']=pd.DataFrame(boston.target)
data


# In[7]:


pd.DataFrame(data.corr().round(2))


# In[8]:


x=data['RM']


# In[9]:


y=data['MEDV']


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


linearRegressionClassifier = LinearRegression()


# In[13]:


x=pd.DataFrame(x)
y=pd.DataFrame(y)


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[15]:


print(x_train.shape)


# In[16]:


linearRegressionClassifier.fit(x_train, y_train)


# In[17]:


import numpy as np


# In[18]:


from sklearn.metrics import mean_squared_error


# In[19]:


y_pred=linearRegressionClassifier.predict(x_test)
y_pred.shape


# In[20]:


np.sqrt(mean_squared_error(y_test, y_pred))


# In[21]:


linearRegressionClassifier.score(x_test,y_test)


# In[ ]:


#working with ridges


# In[22]:


from sklearn.linear_model import Ridge


# In[23]:


ridge1=Ridge(alpha=1)


# In[24]:


ridge1.fit(x_train,y_train)


# In[25]:


y_pred1=ridge1.predict(x_test)


# In[26]:


np.sqrt(mean_squared_error(y_test, y_pred1))


# In[27]:


ridge2=Ridge(alpha=100)


# In[28]:


ridge2.fit(x_train,y_train)


# In[29]:


y_pred2=ridge2.predict(x_test)


# In[30]:


np.sqrt(mean_squared_error(y_test, y_pred2))


# In[31]:


ridge2.score(x_test,y_test)


# In[ ]:


#working with lasso


# In[32]:


from sklearn.linear_model import Lasso


# In[33]:


Lasso1=Lasso(alpha=0.01)
Lasso1.fit(x_train,y_train)


# In[34]:


y_predL1=Lasso1.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_predL1))


# In[35]:


Lasso1.score(x_test,y_test)


# In[ ]:


#working with ElasticNet


# In[36]:


from sklearn.linear_model import ElasticNet


# In[37]:


en1=ElasticNet(alpha=0.1, l1_ratio=0.5)
en1.fit(x_train,y_train)


# In[38]:


y_pred_en1=en1.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred_en1))


# In[39]:


en1.score(x_test,y_test)


# In[ ]:




