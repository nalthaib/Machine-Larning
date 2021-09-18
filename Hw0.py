#!/usr/bin/env python
# coding: utf-8

# In[93]:


# Nasser Althaiban   800764203   HW0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[94]:


df = pd.read_csv('https://raw.githubusercontent.com/nalthaib/Machine-Larning/main/D3.csv')
df.head(99)
#M=len(df)
#M


# In[95]:


X1= df.values[:,0]
X2= df.values[:,1]
X3= df.values[:,2]
Y = df.values[:,3]
m = len(Y)
#print('X1 = ', X1[: 99])
#print('X2 = ', X2[: 99])
#print('X3 = ', X3[: 99])
#print('Y = ', Y[: 99])


# In[96]:


plt.scatter(X1,Y,marker='+')
plt.grid()
plt.rcParams["figure.figsize"] #= (10,6)
plt.xlabel('Variable X1')
plt.ylabel('Output Y')
plt.title('Scatter plot of training data')


# In[97]:


plt.scatter(X2,Y,marker='*')
plt.grid()
plt.rcParams["figure.figsize"] #= (10,6)
plt.xlabel('Variable X2')
plt.ylabel('Output Y')
plt.title('Scatter plot of training data')


# In[99]:


plt.scatter(X3,Y,marker='.')
plt.grid()
plt.rcParams["figure.figsize"] #= (10,6)
plt.xlabel('Variable X3')
plt.ylabel('Output Y')
plt.title('Scatter plot of training data')


# In[100]:


#Lets create a matrix with single column of ones
X_0 = np.ones((m,1))
X_0[:5]


# In[101]:


# Using reshape function convert X 1D array to 2D array of dimension 97x1
X_1 = X1.reshape(m, 1)
X_1[:10]


# In[102]:


# Using reshape function convert X 1D array to 2D array of dimension 97x1
X_2 = X2.reshape(m, 1)
X_2[:10]


# In[103]:


# Using reshape function convert X 1D array to 2D array of dimension 97x1
X_3 = X3.reshape(m, 1)
X_3[:10]


# In[104]:


# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final X matrix (feature matrix)
X1 = np.hstack((X_0, X_1))# , X_2, X_3))
X2 = np.hstack((X_0, X_2))
X3 = np.hstack((X_0, X_3))


# In[105]:


theta = np.zeros(2)
theta


# In[106]:


def compt_cost(X1, Y, theta):

    predictions = X1.dot(theta)
    errors = np.subtract(predictions, Y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m) * np.sum(sqrErrors)
    return J


# In[107]:


# Lets compute the cost for theta values
cost1 = compt_cost(X1, Y, theta)
print('The cost for given values of theta_0 and theta_1 =', cost1)
cost2 = compt_cost(X2, Y, theta)
print('The cost for given values of theta_0 and theta_2 =', cost2)
cost3 = compt_cost(X3, Y, theta)
print('The cost for given values of theta_0 and theta_3 =', cost3)


# In[108]:


def gradient_descent(X1, Y, theta, alpha, iterations):

    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = X1.dot(theta)
        errors = np.subtract(predictions, Y)
        sum_delta = (alpha / m) * X1.transpose().dot(errors);
        theta = theta - sum_delta;
        cost_history[i] = compt_cost(X1, Y, theta)
    return theta, cost_history


# In[112]:


theta = [0., 0.]
iterations = 1500;
alpha = 0.01;


# In[113]:


theta, cost_history = gradient_descent(X1, Y, theta, alpha, iterations)
print('Final value of theta X1 =', theta)
print('cost_history X1 =', cost_history)


# In[114]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(X1[:,1], Y, color='red', marker= '*', label= 'Training Data')
plt.plot(X1[:,1],X1.dot(theta), color='green', label='Linear Regression')
plt.rcParams["figure.figsize"] #= (10,6)
plt.grid()
plt.xlabel('Variable X1')
plt.ylabel('Output Y')
plt.title('Linear Regression Fit')
plt.legend()


# In[115]:


plt.plot(range(1, iterations + 1),cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[116]:


theta, cost_history = gradient_descent(X2, Y, theta, alpha, iterations)
print('Final value of theta X2 =', theta)
print('cost_history X2 =', cost_history)


# In[76]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(X2[:,1], Y, color='red', marker= '*', label= 'Training Data')
plt.plot(X2[:,1],X2.dot(theta), color='green', label='Linear Regression')
plt.rcParams["figure.figsize"] #= (10,6)
plt.grid()
plt.xlabel('Variable X2')
plt.ylabel('Output Y')
plt.title('Linear Regression Fit')
plt.legend()


# In[117]:


plt.plot(range(1, iterations + 1),cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[118]:


theta, cost_history = gradient_descent(X3, Y, theta, alpha, iterations)
print('Final value of theta X3 =', theta)
print('cost_history X3 =', cost_history)


# In[80]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(X3[:,1], Y, color='red', marker= '*', label= 'Training Data')
plt.plot(X3[:,1],X3.dot(theta), color='green', label='Linear Regression')
plt.rcParams["figure.figsize"] #= (10,6)
plt.grid()
plt.xlabel('Variable X3')
plt.ylabel('Output Y')
plt.title('Linear Regression Fit')
plt.legend()


# In[119]:


# Problem 2

X= np.hstack((X_0, X_1 , X_2, X_3))


# In[120]:


theta = np.zeros(4)
theta


# In[121]:


def compt_cost(X, Y, theta):

    predictions = X.dot(theta)
    errors = np.subtract(predictions, Y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m) * np.sum(sqrErrors)
    return J


# In[123]:


def gradient_descent(X, Y, theta, alpha, iterations):

    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, Y)
        sum_delta = (alpha / m) * X.transpose().dot(errors);
        theta = theta - sum_delta;
        cost_history[i] = compt_cost(X, Y, theta)
    return theta, cost_history


# In[124]:


# Lets compute the cost for theta values
cost1 = compt_cost(X, Y, theta)
print('The cost for given values of theta_0 and theta =', cost)


# In[125]:


theta = [0., 0. ,0. ,0.]
iterations = 1500;
alpha = 0.01;


# In[126]:


theta, cost_history = gradient_descent(X, Y, theta, alpha, iterations)
print('Final value of theta X =', theta)
print('cost_history X =', cost_history)


# In[127]:


plt.plot(range(1, iterations + 1),cost_history, color='blue')
plt.rcParams["figure.figsize"] #= (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[128]:


#Finally, output Prediction

Xnew = ([1, 1, 1, 1],
        [1, 2, 0, 4],
        [1, 3, 2, 1])
New_Prediction = np.dot(Xnew, theta)
New_Prediction


# In[ ]:




