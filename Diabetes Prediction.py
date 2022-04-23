#!/usr/bin/env python
# coding: utf-8

# ## Importing the Dependencies

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# pd.read_csv?


# ## Data collection and Analysis
# 
# PIMA Diabetes Dataset

# In[6]:


# Loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('D:\AI\Machine Learning\Projects\Diabetes Prediction\diabetes.csv')


# In[7]:


# Printing the first 5 rows of the dataset
diabetes_dataset.head()


# In[8]:


# Printing the last 5 rows of the dataset
diabetes_dataset.tail()


# In[10]:


# The number of rows and columns in this dataset
diabetes_dataset.shape


# In[11]:


# Getting the statistical measures of the data
diabetes_dataset.describe()


# In[12]:


# How many cases are there for diabetes examples and non-diabetes examples?
diabetes_dataset['Outcome'].value_counts()


# In[14]:


# We can also write the code like following
diabetes_dataset.value_counts('Outcome')


# 0 --> Non-diabetic
# 
# 1 --> Diabetic

# In[21]:


# We my try to get the mean for all those values for this label zero and one
diabetes_dataset.groupby('Outcome').mean()
# It is good practise to always group dataset based on their label


# In[24]:


# Separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']


# In[25]:


print(X)
print(Y)


# ## Data Standardisation

# In[27]:


scaler = StandardScaler()
scaler.fit(X)
standardised_data = scaler.transform(X)


# In[29]:


print(standardised_data)


# In[33]:


# We can also use "scaler.fit_transform(X)"
standardised_data1 = scaler.fit_transform(X)


# In[34]:


print(standardised_data1)


# All these values are in the range of zero and one. Thus, this will help our model to make better prediction because all the values are almost in the similar range.

# In[35]:


X = standardised_data
Y = diabetes_dataset['Outcome']


# In[36]:


print(X)
print(Y)


# ## Train Test Split

# In[45]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 42)


# In[46]:


print(X.shape, X_train.shape, X_test.shape)


# ## Training the model

# In[48]:


classifier = svm.SVC(kernel = 'linear')


# In[49]:


# Training the support Vector Machine Classifier
classifier.fit(X_train, Y_train) # We have to mention the training data which are X_train and the labels for the training 
                                 # data which is Y_train 


# ## Model evaluation
# ### Accuracy Score

# In[51]:


# Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[52]:


print('Accuracy score of the training data: ', training_data_accuracy)


# In[53]:


# Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[54]:


print('Accuracy score of the test data: ', test_data_accuracy)


# ## Making a predictive system

# In[76]:


input_data = (7,159,64,0,0,27.4,0.294,40)
# Changing the input data to numpy array 
input_data_as_numpy_array = np.asarray(input_data)
# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# Standardise the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')

