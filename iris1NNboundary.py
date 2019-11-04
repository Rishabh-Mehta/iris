import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.markers import MarkerStyle
from scipy.spatial.distance import cdist
from scipy import stats
np.random.seed(1)
#  KNN function
def knn_predict(X_test, X_train, y_train, k):
    n_X_test = X_test.shape[0]
    decision = np.zeros((n_X_test, 1))
    for i in range(n_X_test):
        point = X_test[[i],:]

        #  compute euclidan distance from the point to all training data
        dist = cdist(X_train, point)

        #  sort the distance, get the index
        idx_sorted = np.argsort(dist, axis=0)

        #  find the most frequent class among the k nearest neighbour
        pred = stats.mode( y_train[idx_sorted[0:k]] )

        decision[i] = pred[0]
    return decision


# Setup data
D = np.genfromtxt('iris.csv', delimiter=',')
X_train = D[:, 0:2]   # feature
y_train = D[:, -1]    # label


# Setup meshgrid
x1, x2 = np.meshgrid(np.arange(2,5,0.01), np.arange(0,3,0.01))
X12 = np.c_[x1.ravel(), x2.ravel()]



def plot(decision,m):

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    #  plot decisions in the grid
    decision = decision.reshape(x1.shape)
    plt.figure()
    plt.pcolormesh(x1, x2, decision, cmap=cmap_light)

    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, s=25)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    plt.title( str(m)+' lables flipped')

    plt.show()
    

# Compute 1NN decision
#k = 1
#decision1NN = knn_predict(X12, X_train, y_train, k)
#print("1NN decision")
#plot(decision1NN)


def random_flip(n):
   # print("Labels flipped",n)
    rtrain = y_train
    #print("n",n)
    for k in range(n):
        i = np.random.choice(range(len(rtrain)))
       # print(k)
        #print("I",i)
        #print("old",rtrain[i])
        if rtrain[i] == 1:
            rtrain[i] = np.random.choice([2,3])
        elif rtrain[i] == 2:
            rtrain[i] = np.random.choice([1,3])
        else :
            rtrain[i] = np.random.choice([1,2])
        #print("new",rtrain[i])
    
    return rtrain
    

#decision.reshape(X_train.shape)

def error(decision,test):
  #  print(np.unique(train, return_counts=True))
    error =0
    for i in range(len(decision)):
        if (decision[i][0] != test[i]):
            error = error + 1 
    return error

def leave_outerror(Y_train):
    cv_error = []
    for i in range(len(Y_train)):
        ytest=[]
        ytest.append(Y_train[i])
        ytrain = np.delete(Y_train,i,axis=0)
        xtest = X_train[i]
        xtrain = np.delete(X_train,i,axis=0)
        d=knn_predict(xtest.reshape(1,2),xtrain,ytrain,3)
        e = error(d,ytest)
        cv_error.append(e)
    return cv_error
    
   
 
             

m = [10,20,30,50]
for i in range(len(m)):
    #print("Samples changed",random_input[i])
    train = []
    train= random_flip(m[i])
    decision_random = knn_predict(X12,X_train,train,3)
    E_decision=knn_predict(X_train,X_train,train,3)
    T_error = error(E_decision,train)
    print(m[i],"Labels flipped")
    print("Error count",T_error)
    print("Error rate",round((T_error)/len(E_decision)*100,3),"%")
    cv= leave_outerror(train)
    print("LOOCV error for ",m[i],"flips")
    print("Error count",sum(cv))
    print(round(sum(cv)/len(cv)*100,3),"%")
    print("---------------------------------------------")
    #plot(decision_random,m[i])
    