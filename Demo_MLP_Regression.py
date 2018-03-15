
# coding: utf-8

# In[1]:


from MLP_Regression import *
import pandas as pd
import numpy as np
import random
import sklearn.model_selection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# ##### Create toy data

# In[21]:


x1 = []
x2 = []
noise = []
for i in range(1000):
    x1.append(random.uniform(-3,3))
    x2.append(random.uniform(-3,3))
    noise.append(random.uniform(-1, 1))
x1 = np.array(x1)
x2 = np.array(x2)
noise = np.array(noise)
y = (((x1 + x2)**5) / 3000) + noise

d = {'x1': x1, 'x2': x2}
df = pd.DataFrame(data=d)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df, y, test_size=0.25, random_state=101)
print("Creating toy data...")


# ##### Train model and predict values

# In[33]:


MLP = MLP_reg(M = 5, nTrials=5, maxEpochs = 20, nExplore=10, eta=0.005, lbd=0.001, nMicroEpochs=10)
MLP.train(X_train, y_train)
pred_mlp = MLP.predict(X_test)
print("Training model and predicting values for unseen data...")


# In[34]:


print("Sum of residuals (prediction vs. target) as prediction quality measure:")
print(np.abs(pred_mlp - y_test.T).sum())


# ##### Plot predictions vs real target values (y-scale)

# In[35]:


MLP.plot3D(X=np.mat(X_test), T=y_test)

