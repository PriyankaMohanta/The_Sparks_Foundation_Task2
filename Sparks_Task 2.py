#!/usr/bin/env python
# coding: utf-8

# **The Sparks Foundation**

# **Data Science and Business Analytics Internship**

# **Task-2 :Prediction using UnSupervised ML**
# 

# **TASK :From the given ‘Iris’ dataset, predict the optimum number of clusters
# and represent it visually**

# **By-Priyanka Mohanta**

# In[1]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the iris dataset
data=pd.read_csv(r"C:\Users\kabya\Downloads\Iris.csv")


# In[3]:


#rename the iris dataset columns name
data.columns=['ID','SL','SW','PL','PW','Species']


# In[4]:


#check the dataset
data


# In[5]:


#check the shape of the dataset
data.shape


# In[6]:


data.info()


# In[7]:


#summery statistics for numerical columns
data.describe()


# In[8]:


#check the first 5 rows
data.head()


# In[9]:


#check the last 5 rows
data.tail()


# In[10]:


#check the null value of this dataset
data.isnull().sum()


# In[11]:


#Pearson's Correlation
cor=data.corr()
plt.figure(figsize=(10,5))
sns.heatmap(cor,annot=True,cmap='coolwarm')
plt.show()


# In[12]:


ip=data.drop(['Species'],axis=1)

from sklearn import cluster
km=cluster.KMeans(n_clusters=3)
km.fit(ip)
k=km.predict(ip)
print(k)

data['predict']=k

plt.figure(figsize=(12,5))
plt.scatter(data.SL,data.PL)
plt.show()


# In[14]:


#centroid
plt.figure(figsize=(12,5))
plt.scatter(data.SL,data.PL,c=k,s=50,cmap='viridis')
plt.show()
centroid=km.cluster_centers_
print(centroid)


# In[15]:


#find the optimum number of clusters for K Means
x = ip.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[16]:


#visualization graph by using elbow method
plt.plot(range(1, 11),wcss,marker='o',color='g')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# From this we choose the number of clusters as 3

# In[17]:


# Applying kmeans clustering to the dataset 
kmeans = KMeans(n_clusters = 3,max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[19]:


# Visualising the clusters 
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()
plt.show()


# **data_set_of_make_blobs**

# In[20]:


from sklearn.datasets.samples_generator import make_blobs
X,y=make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=0)
data=pd.DataFrame(X,y)
plt.scatter(X[:,0],X[:,1],s=50)
plt.show()


# In[21]:


from sklearn import cluster

km=cluster.KMeans(n_clusters=4)
km.fit(X)

yp=km.predict(X)
data['predict']=yp

plt.scatter(X[:,0],X[:,1],c=yp,s=50,cmap='viridis')
plt.scatter(centroid[:,0],centroid[:,1],c='black',s=200)
plt.show()
centroid=km.cluster_centers_
print(centroid)


# **THANK YOU**

# In[ ]:




