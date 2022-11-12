#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


path = 'D:\\tasks for ml\\data for model.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])


# In[3]:


print('data = \n' ,data.head(10) )
print('**************************************')
print('data.describe = \n',data.describe())
print('**************************************')


# In[4]:


data.plot(kind='scatter', x='Population', y='Profit', figsize=(5,5),c="r")


# In[5]:


# adding a new column called ones before the data
data.insert(0, 'Ones', 1)
print('new data = \n' ,data.head(10) )


# In[6]:


cols = data.shape[1] # 97*3_____>3
X = data.iloc[:,0:cols-1]# take all rows and cols from starting to befroe ending with one 
y = data.iloc[:,cols-1:cols]# take the last colm and all row its own from table
print('X data = \n' ,X.head(10) )# sample for the first 10 elment
print('**************************************')
print('y data = \n' ,y.head(10) )#sample for the first 10 elment


# In[7]:


X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))


# In[15]:


# cost function
def computeCost(X, y, theta):
    z=np.power(((X * theta.T) - y), 2)
   # print('z \n',z)
   # print('m ' ,len(X))
    return np.sum(z) / (2 * len(X))
print('computeCost(X, y, theta) = ' , computeCost(X, y, theta))


# In[27]:


# GD function
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])# int num of theta
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))#perdiction value for theta 
            theta = temp # make the theta is a perdiction value 
            cost[i] = computeCost(X, y, theta)
        return theta, cost


# In[68]:


# initialize variables for learning rate and iterations
alpha = 0.01
iters = 1000
# perform gradient descent to "fit" the model parameters
g, cost = gradientDescent(X, y, theta, alpha, iters)
print('g = ' , g)
print('cost = ' , cost[0:50] )
print('computeCost = ' , computeCost(X, y, g))


# In[69]:


x = np.linspace(data.Population.min(), data.Population.max(), 100)
print('x \n',x)
print('g \n',g)
f = g[0, 0] + (g[0, 1] * x)
print('f \n',f)


# In[70]:


fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')


# In[71]:


fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations') 
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


# In[ ]:




