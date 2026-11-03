import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
# All data sets are loaded from the library
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_california_housing


# -------------------------------------------
# Functions for Distributed Sketching Methods
# -------------------------------------------

# Standard Method 
def OLSs(X,Y,k,m,S,model):
    Yp = []
    for i in range(k):
        # Worker builds OLS estimator based on sketch
        model.fit(S[i]@X, S[i]@Y)
        Yp.append(model.predict(X))
    # Center aggregates local estimators 
    if k > 1: 
        Yp = np.mean(Yp,axis=0)
    else:
        Yp = Yp[0]
    er = mean_squared_error(Y,Yp)
    return er

# Proposed Method 
def OLSps(X,Y,k,m,S,model):
    n,d = X.shape
    p = int(n/k) # size of each partition 
    Yp = []
    for i in range(k):
        idx_head = i*p
        idx_tail = (i+1)*p
        # partition data set 
        Xi = X[idx_head:idx_tail,:]
        Yi = Y[idx_head:idx_tail]
        # truncate sketching matrix
        Si = S[i][:,0:p]
        # Worker builds OLS estimator based on sketch
        model.fit(Si@Xi, Si@Yi)
        Yp.append(model.predict(X))
    # Center aggregates local estimators 
    if k > 1: 
        Yp = np.mean(Yp,axis=0)
    else:
        Yp = Yp[0]
    er = mean_squared_error(Y,Yp)
    return er


# ------------------------------
# Clean Data
# ------------------------------
def DataClean(X):
    # remove constant features 
    a = np.std(X,axis=0)
    idx_rm = np.where(a==0)[0]
    X = np.delete(X, idx_rm, axis=1)
    # remove one linearly dependent features 
    # repeatedly call the function to remove more features 
    n, d = X.shape
    gap = d - np.linalg.matrix_rank(X)
    if gap > 0:
        idx_rm = []
        for i in range(d):
            Xi = np.delete(X, i, axis=1)
            if np.linalg.matrix_rank(Xi) == d-1:
                idx_rm.append(i)
                break
        X = np.delete(X, idx_rm, axis=1)        
    return X


# ------------------------------
# Main Program
# ------------------------------

D = load_digits()
# D = fetch_california_housing()
# D = fetch_covtype() 
X = D.data
Y = D.target
X = X[:100000,:]
Y = Y[:100000]
X = DataClean(X)
n,d = X.shape

# choice of m and k
m = d+2
k_vec = [1,2,3,4,5] 
n_trial = 1000


# ------------------------------
# Baseline Method, no sketching 
# ------------------------------
model = LinearRegression(fit_intercept=False)
model.fit(X, Y)
Yp = model.predict(X)
er0 = mean_squared_error(Y,Yp)


# ----------------------------------
# Two Distributed Sketching Methods 
# 1: standard, OLSs
# 2: proposed, OLSps
# ----------------------------------
er1_k = []
er2_k = []
for k in k_vec:
    er1_trial = []
    er2_trial = []
    # Results averaged over random choices of sketching matrices
    for t in range(n_trial):
        print([k,t])
        # Generate sketching matrices 
        # Used by both methods for fairer comparison 
        # Truncated for the proposed method
        S = []
        for j in range(k):
            S.append(np.random.normal(loc=0, scale=np.sqrt(1/m), size=(m,n)))
        # Run the two methods 
        er1 = OLSs(X,Y,k,m,S,model)
        er2 = OLSps(X,Y,k,m,S,model)
        er1_trial.append((er1-er0)) # normalized excess loss, 
        er2_trial.append((er2-er0)) # multiply n for unnormalized loss 
    er1_k.append(np.mean(er1_trial))
    er2_k.append(np.mean(er2_trial))


# ----------------
# Plot Loss Curve 
# ----------------
marksize = 10
fontsize = 14
plt.figure()
plt1, = plt.plot(k_vec,er1_k, label='$\\bar{\\beta}$', marker='s',markersize=marksize, markerfacecolor="None")
plt2, = plt.plot(k_vec,er2_k, label='$\\bar{\\theta}$', marker='d',markersize=marksize, markerfacecolor="None")
plt.xlabel('k')
plt.ylabel('Expected Excess Loss')
plt.xticks(k_vec)
plt.legend(loc='upper right',fontsize=fontsize)
plt.show()






