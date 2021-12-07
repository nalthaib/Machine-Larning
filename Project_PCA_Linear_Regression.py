#!/usr/bin/env python
# coding: utf-8

# In[1263]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[1264]:


weather_solar_energy = pd.DataFrame(pd.read_csv("https://raw.githubusercontent.com/mrahma15/MyRepos/main/Hourly%20Weather%20and%20Solar%20Energy%20Dataset.csv"))
weather_solar_energy.head()


# In[1265]:


m = len(weather_solar_energy)
m


# In[1266]:


weather_solar_energy.shape


# In[1267]:


#Splitting the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# We specify random seed so that the train and test data set always have the same rows, respectively
#np.random.seed(0)
df_train, df_test = train_test_split(weather_solar_energy, train_size = 0.7, test_size = 0.3)

df_train.shape


# In[1268]:


df_test.shape


# In[1269]:


num_vars = ['Cloud coverage', 'Visibility', 'Temperature', 'Dew point', 'Relative humidity', 'Wind speed', 'Station pressure', 'Altimeter', 'Solar energy']
df_Newtrain = df_train[num_vars]
df_Newtest = df_test[num_vars]
df_Newtrain.head()


# In[1270]:


df_Newtrain.shape


# In[1271]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# define standard scaler
#scaler = StandardScaler()
scaler = MinMaxScaler()
df_Newtrain[num_vars] = scaler.fit_transform(df_Newtrain[num_vars])
df_Newtrain.head(20)


# In[1272]:


df_Newtest[num_vars] = scaler.fit_transform(df_Newtest[num_vars])
df_Newtest.head(20)


# In[1273]:


y_Newtrain = df_Newtrain.pop('Solar energy')
X_Newtrain = df_Newtrain


# In[1274]:


X_Newtrain.head()


# In[1275]:


y_Newtrain.head()


# In[1276]:


y = y_Newtrain.values

print('y = ', y[: 5])


# In[1277]:


#importing Principal Component Analysis

from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pc_train = pca.fit_transform(X_Newtrain)
pc_train = pd.DataFrame(pc_train)
pc_train


# In[1278]:


# getting the input values from each column and putting them as a separate variable for training set

X1 = pc_train.values[:, 0]               
X2 = pc_train.values[:, 1]               
X3 = pc_train.values[:, 2] 
X4 = pc_train.values[:, 3]
X5 = pc_train.values[:, 4]
             
print('X1 = ', X1[: 5]) 
print('X2 = ', X2[: 5])
print('X3 = ', X3[: 5])


# In[1279]:


m = len(pc_train)               # size of training set
X_0 = np.ones((m, 1))             # Creating a matrix of single column of ones as X0 with the size of training set
X_0 [: 5]


# In[1280]:


# Converting 1D arrays of training X's to 2D arrays

X_1 = X1.reshape(m, 1)
X_2 = X2.reshape(m, 1)
X_3 = X3.reshape(m, 1)
X_4 = X4.reshape(m, 1)
X_5 = X5.reshape(m, 1)


print('X_1 = ', X_1[: 5])
print('X_2 = ', X_2[: 5])
print('X_3 = ', X_3[: 5])


# In[1281]:


# Stacking X_0 through X_11 horizotally
# This is the final X Matrix for training

X = np.hstack((X_0, X_1, X_2, X_3, X_4, X_5))
X [: 5]


# In[1282]:


theta = np.zeros(6)
theta


# In[1283]:


y_Newtest = df_Newtest.pop('Solar energy')
X_Newtest = df_Newtest


# In[1284]:


X_Newtest.head()


# In[1285]:


y_Newtest.head()


# In[1286]:


y_test = y_Newtest.values

print('y_test = ', y_test[: 5])


# In[1287]:


pc_test = pca.fit_transform(X_Newtest)
pc_test = pd.DataFrame(pc_test)
pc_test


# In[1288]:


# getting the input values from each column and putting them as a separate variable for validation set

X1_test = pc_test.values[:, 0]                
X2_test = pc_test.values[:, 1]                
X3_test = pc_test.values[:, 2]
X4_test = pc_test.values[:, 3]
X5_test = pc_test.values[:, 4]


# In[1289]:


m_test = len(pc_test)                # size of validation set
X_0_test = np.ones((m_test, 1))        # Creating a matrix of single column of ones as X0 with the size of validation set


# In[1290]:


# Converting 1D arrays of validation X's to 2D arrays

X_1_test = X1_test.reshape(m_test, 1)
X_2_test = X2_test.reshape(m_test, 1)
X_3_test = X3_test.reshape(m_test, 1)
X_4_test = X4_test.reshape(m_test, 1)
X_5_test = X5_test.reshape(m_test, 1)


# In[1291]:


# Stacking X_0_test through X_11_test horizotally
# This is the final X Matrix for validation

X_test = np.hstack((X_0_test, X_1_test, X_2_test, X_3_test, X_4_test, X_5_test))
X_test [: 5]


# In[1292]:


# defining function for computing the cost 

def compute_cost(X, y, theta, m):
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m) * np.sum(sqrErrors)
    return J


# In[1293]:


# defining function for gradient descent algorithm
# gradient descent algorithm is applied on the training set
# for each iteration loss for both training and validation set is calculated

def gradient_descent(X, y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    cost_test = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * X.transpose().dot(errors);
        theta = theta - sum_delta;
        cost_history[i] = compute_cost(X, y, theta, m)                       # loss for training set
        cost_test[i] = compute_cost(X_test, y_test, theta, m_test)           # loss for training set
     
    return theta, cost_history, cost_test


# In[1294]:


# computing the cost for initial theta values

cost = compute_cost(X, y, theta, m)
cost


# In[1295]:


theta = [0., 0., 0., 0., 0., 0.]
iterations = 1000;
alpha = 0.1


# In[1296]:


# Computing final theta values and losses for training and validation set

theta, cost_history, cost_test = gradient_descent(X, y, theta, alpha, iterations)
print('Final value of theta=', theta)
print('cost_history =', cost_history)
print('cost_test =', cost_test)


# In[1297]:


plt.plot(range(1, iterations + 1),cost_history, color='blue', label= 'Loss for Training Set')
plt.plot(range(1, iterations + 1),cost_test, color='red', label= 'Loss for Evaluation Set')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[ ]:




