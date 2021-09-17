#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/nalthaib/Machine-Larning/main/D3.csv')
df.head(99)
#M=len(df)
#M


# In[44]:


X1= df.values[:,0]
X2= df.values[:,1]
X3= df.values[:,2]
Y = df.values[:,3]
m = len(Y)
#print('X1 = ', X1[: 99])
#print('X2 = ', X2[: 99])
#print('X3 = ', X3[: 99])
#print('Y = ', Y[: 99])


# In[45]:


plt.scatter(X1,Y,marker='+')
plt.grid()
plt.rcParams["figure.figsize"] #= (10,6)
plt.xlabel('Variable X1')
plt.ylabel('Output Y')
plt.title('Scatter plot of training data')


# In[58]:


plt.scatter(X2,Y,marker='*')
plt.grid()
plt.rcParams["figure.figsize"] #= (10,6)
plt.xlabel('Variable X2')
plt.ylabel('Output Y')
plt.title('Scatter plot of training data')


# In[59]:


plt.scatter(X3,Y,marker='.')
plt.grid()
plt.rcParams["figure.figsize"] #= (10,6)
plt.xlabel('Variable X3')
plt.ylabel('Output Y')
plt.title('Scatter plot of training data')


# In[48]:


#Lets create a matrix with single column of ones
X_0 = np.ones((m, 1))
X_0[:5]


# In[49]:


# Using reshape function convert X 1D array to 2D array of dimension 97x1
X_1 = X1.reshape(m, 1)
X_1[:10]


# In[50]:


# Using reshape function convert X 1D array to 2D array of dimension 97x1
X_2 = X2.reshape(m, 1)
X_2[:10]


# In[51]:


# Using reshape function convert X 1D array to 2D array of dimension 97x1
X_3 = X3.reshape(m, 1)
X_3[:10]


# In[52]:


# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final X matrix (feature matrix)
X = np.hstack((X_0, X_1))# , X_2, X_3))
X[:5]


# In[53]:


theta = np.zeros(2)
theta


# In[54]:


def compt_cost(X, Y, theta):

predictions = X.dot(theta)
errors = np.subtract(predictions, Y)
sqrErrors = np.square(errors)
J = 1 / (2 * m) * np.sum(sqrErrors)
return J


# In[35]:


# Lets compute the cost for theta values
cost = compt_cost(X, Y, theta)
print('The cost for given values of theta_0 and theta_1 =', cost)


# In[57]:


def gradient_descent(X, y, theta, alpha, iterations):

cost_history = np.zeros(iterations)
for i in range(iterations):
predictions = X.dot(theta)
errors = np.subtract(predictions, y)
sum_delta = (alpha / m) * X.transpose().dot(errors);
theta = theta - sum_delta;
cost_history[i] = compute_cost(X, y, theta)
return theta, cost_history


# In[55]:


theta = [0., 0.]
iterations = 1500;
alpha = 0.01;


# In[39]:


theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
print('Final value of theta =', theta)
print('cost_history =', cost_history)


# In[40]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(X[:,1], y, color='red', marker= '+', label= 'Training Data')
plt.plot(X[:,1],X.dot(theta), color='green', label='Linear Regression')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Linear Regression Fit')
plt.legend()


# In[41]:


plt.plot(range(1, iterations + 1),cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[ ]:




