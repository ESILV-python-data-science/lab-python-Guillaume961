
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np


# In[13]:


jfk_df = pd.read_csv('jfkrelease-2017-dce65d0ec70a54d5744de17d280f3ad2.csv', sep=';')


# In[51]:


#jfk_df.head()


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

