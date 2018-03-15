import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MLP_reg:
    
    """Builds MLP regression object with one hidden layer and M hidden nodes. Use self.train() to train model with parameters below. Use self.predict() to predict target values for unseen input data. Use self.plot3D() to plot data points and a wireframe of predicted target values (works only for data with 2 feature vectors and 1 target vector).

    Parameters
    ---------
    M: int, no. of hidden nodes (set high for learning complex relations, low for fast learning)
    eta: float, learning rate (set high for fast learning, low for precise learning (more epochs required))
    eta_inc: float, factor by which learning rate increases if model error decreased after a learning epoch in finalization mode (adaptive learning rate)
    eta_dec: float, factor by which learning rate decreases if model error increased after a learning epoch in finalization mode (adaptive learning rate)
    eta_fade: float, factor by which learning rate decreases in exploration mode
    maxEpochs: int, number of times the algorithm is iterating over the training data (learning epochs)
    nMicroEpochs: int, splits training data into n evenly-sized batches (set high for fast learning, low for precise learning)
    nExplore: int, number of learning epochs in exploration mode; switching to finalization mode afterwards (longer exploration phase gives higher propability of finding the global minimum of the model error function)
    lbd: float, factor for regularization term (set high to avoid overfitting but prediction accuracy is lost if set too high)
    
    """
    
    def __init__(self, M=3, eta=0.01, eta_inc=1.05, eta_dec=0.7, eta_fade=0, maxEpochs=100, nMicroEpochs=1, nExplore=50, nTrials=5, lbd=0.001):
        self.M = M
        self.eta = eta
        self.eta_inc = eta_inc
        self.eta_dec = eta_dec
        self.eta_fade = eta_fade
        self.nMicroEpochs = nMicroEpochs
        self.maxEpochs = maxEpochs
        self.nExplore = nExplore
        self.nTrials = nTrials
        self.lbd = lbd
        self.X = None
        self.T = None
        self.N = None
        self.D = None
        self.K = None
        self.W1 = None
        self.W2 = None
        self.a = None
        self.y = None
        self.z = None
        self.E = None
    
    def setTrainingData(self, X, T):
        #Matrix X with data vectors as rows (data/activation vectors always as row vectors below!!!)
        self.X = np.mat(X)
        X = np.concatenate((np.ones((len(X),1)),X),1)
        self.X = np.mat(X)
        self.N, self.D = np.shape(X)
        self.T = np.matrix(T)
        if(len(self.T) != self.N): self.T = self.T.T
        if np.shape(self.T)[1] != 1: return("Target data contains multiple columns; only one allowed!")
        self.K = len(self.T.T)
    
    def getRandomWeights(self): #Function redundant; use setRandomWeights
        #Matrices with weight vectors as columns of length starting nodes (i.e. length D or M) 
        W1 = np.mat(np.random.rand(self.D, self.M))
        W2 = np.mat(np.random.rand(self.M, self.K))
        return W1, W2

    def setRandomWeights(self):
        #Matrices with weight vectors as columns of length starting nodes (i.e. length D or M) 
        self.W1 = np.mat(np.random.rand(self.D, self.M))
        self.W2 = np.mat(np.random.rand(self.M, self.K))

    def propagateActivity(self, x, W1=None, W2=None):
        if W1 is None: W1 = self.W1
        if W2 is None: W2 = self.W2
        self.a = x * W1
        self.z = np.tanh(self.a) #activations of hidden units as row vector
        self.y = self.z * W2 
        return self.y
    
    def getError(self, W1=None, W2=None, idx=None):
        if W1  is None: W1 = self.W1
        if W2  is None: W2 = self.W2
        if idx is None: idx = range(len(self.X))
        E = 0 #initialize model error
        for n in idx:
            y = self.propagateActivity(self.X[n], W1, W2) #compute prediction
            E += (np.power(y - self.T[n], 2)) #add squared residual to model error
        reg = self.lbd * (np.power(W1, 2).sum() + np.power(W2, 2).sum()) #calculate regularization term
        E = 0.5 * (E + len(idx)*reg) #add reg term (weighted by batch length) to model error
        return E
    
    def getGradientPart_backprop(self, xn, tn, W1=None, W2=None):
        if W1 is None: W1 = self.W1
        if W2 is None: W2 = self.W2
        self.propagateActivity(xn, W1, W2)
        delta_out = (self.y - tn).item(0)
        delta_hidden = np.multiply((1-np.power(self.z, 2)).T, (W2 * delta_out)) #column vector of hidden nodes errors
        #print(delta_out)
        nabla1n = np.mat(delta_hidden * xn).T + self.lbd*W1 #gradient of weight vectors in hidden layer as M column vectors of length D
        nabla2n = np.mat(self.z * delta_out).T + self.lbd*W2 #gradient(s) of weight vector in output layer as y column vector(s) of length M
        #print(nabla1n, nabla2n)
        return nabla1n, nabla2n
    
    def doLearningEpoch(self, W1=None, W2=None, eta=None): #implementation not efficient (concatenate, shuffle)
        if W1  is None: W1 = self.W1
        if W2  is None: W2 = self.W2
        if eta is None: eta = self.eta
        idxME = np.concatenate((self.X, self.T), axis=1) #concatenate data and target vectors
        np.random.shuffle(idxME) #shuffling data/target vectors
        idxME = np.array_split(idxME, self.nMicroEpochs) #splitting data/target vectors into nMicr. evenly-sized batches
        nabla1, nabla2 = np.zeros(np.shape(W1)), np.zeros(np.shape(W2)) #initialize gradient matrices
        for me in range(self.nMicroEpochs): #iterate over micro epochs
            for xt in idxME[me]: #iterate over data/target vecs in current micro epoch
                v1, v2 = self.getGradientPart_backprop(xt[0,:self.D], xt[0,self.D:], W1, W2) #get gradients for batch
                nabla1 += v1 #update gradients
                nabla2 += v2
            #print(W1, nabla1)
            W1 -= eta * nabla1 #update weights
            W2 -= eta * nabla2
        return W1, W2, self.getError(W1, W2)
    
    def doLearningTrial(self):
        W1, W2 = self.getRandomWeights()
        E = self.getError(W1, W2) #Initial model error with random weights
        eta = self.eta
        epoch = 0
        while epoch < self.nExplore: #Exploration mode (for first nExplore epochs) with monotonic decrease of learning rate
            #print("Explore in epoch: " + str(epoch))
            W1, W2, E = self.doLearningEpoch(W1, W2, eta)
            eta = eta / (1 + self.eta_fade*epoch)
            epoch += 1
        while epoch < self.maxEpochs: #Finalization mode with adaptive learning rate
            W1_new, W2_new, E_new = self.doLearningEpoch(W1, W2, eta)
            if E_new < E: #if error has decreased: Save new weights and error; decrease learning rate
                #print("Finalize with decreased error in epoch: " + str(epoch), str(eta))
                W1, W2, E = W1_new, W2_new, E_new
                eta = eta * self.eta_inc
            else: #if error has increased: increase learning rate
                #print("Finalize with increased error in epoch: " + str(epoch), str(eta))
                eta = eta * self.eta_dec
            epoch += 1
        return W1, W2, E
        
    def train(self, X, T):
        """Train MLP weights
    
        Parameters
        ---------
        X: feature vectors, array-like
            2-dimensional array
        T: target vector, array-like
            1-dimensional array
        
        Sets object attributes
        -------
        self.W1: weights for layer one
        self.W2: weights for layer two
        self.E: model error
        """
        self.setTrainingData(X=X, T=T)
        self.setRandomWeights()
        self.E = self.getError()
        for i in range(self.nTrials):
            W1, W2, E = self.doLearningTrial()
            if E < self.E:
                self.W1, self.W2, self.E = W1, W2, E
                #print("Learning trial no.: " + str(i))    
      
    def predict(self, X):
        """Predict values for unseen data
    
        Parameter
        ---------
        X: feature vectors, array-like
            2-dimensional array
        
        Returns
        -------
        array of predicted values
        """
        #Form input data to matrix adding x_0=1 for bias parameter w_0:
        X = np.mat(X)
        X = np.concatenate((np.ones((len(X),1)),X),1)
        X = np.mat(X)
        #Create and fill array with class predictions
        cls_preds = np.zeros(len(X))
        for i in range(len(X)):
            cls = self.propagateActivity(X[i])[0].item(0)
            cls_preds[i] = cls  
        return cls_preds
        
    def plot3D(self, X=None, T=None):
        """3D plot (works only for 2 feature vectors) including wireframe of predicted values. If no input data is given the training data is used as a default.
    
        Parameter
        ---------
        X: feature vectors, array-like
            2-dimensional array
        T: target vector, array-like
            1-dimensional array
      
        """
        if self.X is None:
            return ("Train model first!")
        if self.T is None:
            return ("Train model first!")
        if self.D != 3:
            return ("Plotting function works only for 3D data (2 predictors + 1 target)!")
        #Transform/create data:
        if X is None: X = self.X[:,1:] 
        if T is None: T = self.T
        if np.shape(X)[1]!=2:
            return ("Plotting function works only for 3D data (2 predictors + 1 target)!")
        X = np.matrix(X)
        T = np.matrix(T)
        if len(T) != len(X): T = T.T
        left = np.min(X[:,0])*1.1
        right = np.max(X[:,0])*1.1
        bottom = np.min(X[:,1])*1.1
        top = np.max(X[:,1])*1.1
        X1,X2 = np.linspace(left, right, 17), np.linspace(bottom, top, 17) #values on x1/x2-axis...
        X1,X2 = np.meshgrid(X1, X2)                          # ...as a meshgrid
        n, m = np.shape(X1)
        y = np.zeros(n*m).reshape(n,m) #initialise 2D-array for predicted values
        for i in range(n): #iterate over X1
            for j in range(m): #iterate over X2
                y[i,j] = self.propagateActivity(np.array([1, X1[i,j], X2[i,j]])) #predict values, store in 2D-array
        
        #Plot data:
        fig = plt.figure(figsize=(10,10))                     # create figure
        ax = fig.add_subplot(111, projection='3d')           # get 3d-axis handle (111: 3 Dimensions have equal scale)
        ax.plot_wireframe(X1, X2, y, rstride=1, cstride=1, color='r', label='Predicted values')   # plot wireframe of regression function
        ax.set_title('True target values vs. wireframe of predicted values \n', fontsize=15)
        ax.scatter(X[:,0],X[:,1],T, c='m', marker='*', s=20, alpha=0.7)     # plot data points
        ax.set_xlabel('x1')                                  # label on x1-axis
        ax.set_ylabel('x2')                                  # label on x2-axis
        ax.set_zlabel('y')                                   # label on y-axis
        plt.show()   


# In[117]:


#Test data and labels
#train = np.array([[-2, -1], [-2, 2], [-1.5, 1], [0, 2], [2, 1], [3, 0], [4, -1], [4, 2],    [-1, -2], [-0.5, -1], [0, 0.5], [0.5, -2], [1, 0.5], [2, -1], [3, -2]])
#labels = np.array([1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1])
#labels2 = np.array(([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1])).T
#labels3 = np.array([1,1,1,1,1,1,1,1,2,2,2,2,3,3,3])


# In[155]:


#MLP_two = MLP_reg(nTrials=1, maxEpochs=1000, nExplore=500, eta=0.01, lbd=0.1)
#MLP_two.train(X=train, T=labels)
#MLP_two.plot3D()

