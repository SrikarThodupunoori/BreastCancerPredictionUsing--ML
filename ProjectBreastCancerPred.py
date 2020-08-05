
# coding: utf-8

# In[21]:


#Importing Libraries
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt 
import seaborn as sns


# In[57]:


#Importing the dataset
df=pd.read_csv('dataset.csv')


# In[68]:


#we can see class/feature of our data
df.columns


# In[69]:


#id and unnamed32 is unnecessary for our model and we change M(malignant) and B(benign) replace 0,1
df.head()


# In[24]:


df.shape


# In[25]:


df.isna().sum()


# In[26]:


df = df.dropna(axis=1)


# In[27]:


df.shape


# In[28]:


#Get a count of the number of 'M'-Malignant & 'B'-Benign cells
df['diagnosis'].value_counts()


# In[29]:


#Visualize this count 
sns.countplot(df['diagnosis'],label="Count")


# In[31]:


#Datatypes in the taken datset
df.dtypes


# In[32]:


#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))


# In[39]:


df.head(5)


# Now I am done exploring and cleaning the data. I will set up my data for the model by first splitting the data set into a feature data set also known as the independent data set (X), and a target data set also known as the dependent data set (Y).

# In[72]:


X = df.iloc[:, 2:31].values 
Y = df.iloc[:, 1].values 


# In[42]:


#we separate train and test data with sklearn selection model
#You can thnk this x_train for learn and y_train is answer of x_train and finally we testing our data with x_test andy_test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[43]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[63]:


print("xtrain:{}".format((x_train).shape))
print("y_train:{}".format((y_train).shape))
print("xtest:{}".format((x_test).shape))
print("ytest:{}".format((y_test).shape))


# In[47]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)


# In[48]:


Y_pred = classifier.predict(X_test)


# In[49]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# In[50]:


cm


# To check the correct prediction we have to check confusion matrix object and add the predicted results diagonally which will be number of correct prediction and then divide by total number of predictions.From the confusion matrix, there are 4 cases of mis-classification. The performance of this algorithm is expected to be high given the symptoms for breast cancer should exchibit certain clear patterns.

# In[70]:


#We draw heatmap for showing confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns
f,ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm,annot = True,linewidth = 1,fmt =".0f",ax = ax)


# The details of the confusion matrix
# 
# True Positive (TP)(cm[0][0])  : Observation is positive, and is predicted to be positive.
# 
# False Negative (FN)(cm[0][1]) : Observation is positive, but is predicted negative.
# 
# True Negative (TN)(cm[1][0])  : Observation is negative, and is predicted to be negative.
# 
# False Positive (FP)(cm[1][1]) : Observation is negative, but is predicted positive.
# 
# Accuracy=(TP+TN)/(TP+FN+TN+FP)

# In[71]:


print(classifier.score(X_train, Y_train))


# This algorithm shows 99% accuracy for the predicted data
