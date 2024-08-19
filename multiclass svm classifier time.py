#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np  
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


irisdata = pd.read_csv('C:/saravanan/E folder/Boxing data coll video/Punch with glove/Jab/feattimepyth.csv')


# In[3]:


irisdata.head()
irisdata.info()


# In[4]:


from sklearn.model_selection import train_test_split
X = irisdata.drop('class', axis=1)  
y = irisdata['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[5]:


kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']#A function which returns the corresponding SVC model
def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return SVC(kernel='linear', gamma="auto")


# In[6]:


for i in range(4):
    # Separate data into test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)# Train a SVC model using different kernal
    svclassifier = getClassifier(i) 
    svclassifier.fit(X_train, y_train)# Make prediction
    y_pred = svclassifier.predict(X_test)# Evaluate our model
    print("Evaluation:", kernels[i], "kernel")
    print(classification_report(y_test,y_pred))


# In[7]:


from sklearn.model_selection import GridSearchCV


# In[8]:


param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}


# In[9]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


# In[10]:


print(grid.best_estimator_)


# In[11]:


grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))


# In[14]:


plt.hist(grid_predictions1)
plt.show() 


# In[14]:


xx = pd.read_csv('C:/saravanan/E folder/Boxing data coll video/Punch with glove/Testing/Testingtimepyth.csv')
yy = pd.read_csv('C:/saravanan/E folder/Boxing data coll video/Punch with glove/Testing/Testing outputpyth.csv')
grid_predictions1 = grid.predict(xx)
print(confusion_matrix(yy,grid_predictions1))
print(classification_report(yy,grid_predictions1))
print(grid_predictions1)
print(yy)


# In[15]:


plt.hist(grid_predictions1)
plt.show() 


# In[13]:


xx1 = pd.read_csv('C:/saravanan/E folder/Boxing data coll video/Punch with glove/Testing/sriram Testingtimepyth.csv')
yy1 = pd.read_csv('C:/saravanan/E folder/Boxing data coll video/Punch with glove/Testing/Testing outputpyth1.csv')
grid_predictions2 = grid.predict(xx1)
print(confusion_matrix(yy1,grid_predictions2))
print(classification_report(yy1,grid_predictions2))
print(grid_predictions2)


# In[ ]:




