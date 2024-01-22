#!/usr/bin/env python
# coding: utf-8

# # Financial Market News - Sentiment Analysis
# 
# This is a data (dummy) of Financial Market Top 25 News for the Day and Task is to Train and Predict Model for Overall Sentiment Analysis
# 

# # Import Library
# 

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# # Import Dataset

# In[5]:


df = pd.read_csv(r'https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Financial%20Market%20News.csv', encoding = "ISO-8859-1")


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.shape


# In[9]:


df.columns


# # Get Feature Selection

# In[10]:


' '.join(str(x) for x in df.iloc[1,2:27])


# In[11]:


df.index


# In[12]:


len(df.index)


# In[13]:


news = []
for row in range(0,len(df.index)):
 news.append(' '.join(str(x) for x in df.iloc[row,2:27]))


# In[14]:


type(news)


# In[15]:


news[0]


# In[16]:


X = news


# In[17]:


type(X)


# # Get Feature Text Conversion to Bag of Words

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer


# In[19]:


cv = CountVectorizer(lowercase = True, ngram_range=(1,1))


# In[20]:


X = cv.fit_transform(X)


# In[21]:


X.shape


# In[22]:


y = df['Label']


# In[23]:


y.shape


# # Get Train Test Split

# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 2529)


# In[26]:


from sklearn.ensemble import RandomForestClassifier


# In[27]:


rf = RandomForestClassifier(n_estimators=200)


# In[28]:


rf.fit(X_train, y_train)


# In[29]:


y_pred = rf.predict(X_test)


# In[30]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[31]:


confusion_matrix(y_test, y_pred)


# In[32]:


print(classification_report(y_test, y_pred))


# In[ ]:




