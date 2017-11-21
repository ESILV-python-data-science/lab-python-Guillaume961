
# coding: utf-8

# In[83]:


import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[13]:


jfk_df = pd.read_csv('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv', sep=';')


# In[57]:


jfk_df.head()


# In[26]:


pageNbAverage = jfk_df['Num Pages'].mean()
pageNbAverage


# In[27]:


pageNbMin = jfk_df['Num Pages'].min()
pageNbMin


# In[28]:


pageNbMax = jfk_df['Num Pages'].max()
pageNbMax


# In[48]:


missingPageNb = jfk_df['Num Pages'].isnull().sum()
missingPageNb


# In[55]:


nbOfDocType =  jfk_df['Doc Type'].nunique()
nbOfDocType


# In[56]:


typesFrequency = jfk_df['Doc Type'].value_counts()
typesFrequency


# In[58]:


nbOfAgencies =  jfk_df['Agency'].nunique()
nbOfAgencies


# In[59]:


agencyFrequency = jfk_df['Agency'].value_counts()
agencyFrequency


# In[92]:


docTypePlot = sns.countplot(x="Doc Type", data=jfk_df)
docTypePlot


# In[99]:


agencyPlot = sns.countplot(x="Agency", data=jfk_df)
agencyPlot


# In[123]:


agencyPlot = sns.countplot(x="Agency", data=jfk_df).semilogy()
agencyPlot


# In[ ]:


# x = np.genfromtxt(jfk_df, delimiter=',', skip_header=10, skip_footer=0, names=['Doc Date']) 
# xdf = pandas.DataFrame(x)
# ax = xdf.plot(x='Doc Date', y='Doc Type')
# ydf.plot(x='TimeStamp', y='Value', ax=ax)

