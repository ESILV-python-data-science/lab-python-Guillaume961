# -*- coding: utf-8 -*-
"""
Classify text
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

file_list = []
column_names = ['text', 'category']
dataframe = pd.read_csv('LeMonde2003.csv', header=None, names=column_names, sep='\t')

dataframe.dropna(axis=0, how='any')

categories = ['ENT','INT','ART','SOC','FRA','SPO','LIV','TEL','UNE']

dataframe = dataframe[dataframe['category'].isin(categories)]

print(dataframe)

sns.countplot(x='category', data=dataframe, palette="Greens_d")
plt.show()

train, dev, test = np.split(dataframe.sample(frac=1), [int(.6*len(dataframe)), int(.8*len(dataframe))]) # 60%, 20%, 20%

vectorizer = CountVectorizer(max_features=1000)
vectorizer.fit(train['text'])
X_train_counts = vectorizer.transform(train['text'])
X_test_counts = vectorizer.transform(test['text'])
X_dev_counts = vectorizer.transform(dev['text'])

print(X_train_counts);
print(X_test_counts);
print(X_dev_counts);


