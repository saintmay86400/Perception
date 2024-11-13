# 1 Linear SVM
# For a binary classification problem, consider a dataset with n samples and m features. 
# The goal is to find a hyperplane defined by: w*x-b=0
# where w is the weight vector and b is the bias term.

# 2 Optimization Prblem
# To find the optimal hyperplane, we need to solve the following optimization problem:
# min 1/2*||w||^2 subject to: yi*(w*xi-b)>= 1 for each i  yi -> class label of the i-th sample

# Kernel Trick
# For non-linear data, the kernel trick is used to map the input features 
# into a higher-dimensional space where a linear separator can be found. 
# Common kernels include the polynomial kernel, radial basis function (RBF) kernel, and sigmoid kernel.

# Step 1: Data Preparation
# First, we need to prepare the dataset. For simplicity, let's use a synthetic dataset.

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
# X:[samples,features] y: [labels] es. 100010101
X, y = make_blobs(n_samples=100, centers=2, random_state=42)
y = np.where(y == 0, -1, 1) # convert labels to -1 and 1

#scatter plot
#plt.scatter(X[:, 0], X[:,1], c=y, cmap='bwr')
#plt.show()

# Step 2: SVM Class Definition
# We'll define SVM class with methods for training and predicting

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w= None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx]*(np.dot(x_i, self.w)-self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
                    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
    
# Step 3: Training the Model on synthetic dataset
# Initialize and train the SVM
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X,y)

# Plot the decision boundary
def plot_decision_boundary(X, y, model):
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr')
    #gca(get current axes) to customize the graphic
    ax = plt.gca()
    #(xmin, xmax)
    xlim = ax.get_xlim()
    # (ymin, ymax)
    ylim = ax.get_ylim()
    
    #linspace(start, stop, num) generate a array 1D of values ​​equidistant between start and stop
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = model.predict(xy).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.show()
    
#plot_decision_boundary(X, y, svm)

# Step 4: Prediction
# To predict the class of new samples, we use the predict method of our SVM class.

new_samples = np.array([[0,0],[4,4]])
predictions = svm.predict(new_samples)
#print(predictions)

#Step 5: Testing the implementation
# To validate our implementation, we can compare it with the SVM implementation from popular libraries such as Scikit-learn.

from sklearn.svm import SVC
# Train SVM using Scikit-learn
clf = SVC(kernel='linear')
clf.fit(X, y)

# Compare predictions
#print(clf.predict(new_samples))