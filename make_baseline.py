
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

train_df = pd.read_pickle("soma_goods_train.df")


# In[3]:

train_df.shape


# In[4]:

import sys


# In[5]:

reload(sys)


# In[6]:

sys.setdefaultencoding('utf8')


# In[7]:

train_df.shape


# In[8]:

sys.getdefaultencoding()


# In[9]:

train_df.shape


# In[10]:

train_df


# In[11]:

from sklearn.feature_extraction.text import CountVectorizer


# In[12]:

vectorizer = CountVectorizer()


# In[13]:

d_list = []
cate_list = []
for each in train_df.iterrows():
    cate = ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])
    d_list.append(each[1]['name'])
    cate_list.append(cate)


# In[14]:

print len(set(cate_list))


# In[15]:

cate_dict = dict(zip(list(set(cate_list)), range(len(set(cate_list)))))


# In[16]:

print cate_dict[u'디지털/가전;네트워크장비;KVM스위치']
print cate_dict[u'패션의류;남성의류;정장']


# In[17]:

y_list = []
for each in train_df.iterrows():
    cate = ";".join([each[1]['cate1'],each[1]['cate2'],each[1]['cate3']])
    y_list.append(cate_dict[cate])


# In[18]:

x_list = vectorizer.fit_transform(d_list)


# In[19]:

from sklearn.svm import LinearSVC


# In[20]:

from sklearn.grid_search import GridSearchCV


# In[21]:

import numpy as np


# In[22]:

svc_param = {'C':np.logspace(-2,0,20)}


# In[23]:

gs_svc = GridSearchCV(LinearSVC(loss='12'), scv_param, cv=5, n_jobs=4)


# In[24]:

gs_svc = GridSearchCV(LinearSVC(loss='12'), svc_param, cv=5, n_jobs=4)


# In[25]:

gs_svc.fit(x_list, y_list)


# In[26]:

gs_svc = GridSearchCV(LinearSVC(loss='12', penalty='l2', dual=True), svc_param, cv=5, n_jobs=4)


# In[27]:

gs_svc.fit(x_list, y_list)


# In[28]:

gs_svc = GridSearchCV(LinearSVC(loss='l2'),svc_param,cv=5,n_jobs=4)


# In[29]:

gs_svc.fit(x_list, y_list)


# In[30]:

print gs_svc.best_params_, gs_svc.best_score_


# In[31]:

clf = LinearSVC(C=gs_svc.best_params_['C'])


# In[32]:

clf.fit(x_list,y_list)


# In[33]:

from sklearn.externals import joblib


# In[34]:

joblib.dump(clf,'classify.model',compress=3)
joblib.dump(cate_dict,'cate_dict.dat',compress=3)
joblib.dump(vectorizer,'vectorizer.dat',compress=3)





