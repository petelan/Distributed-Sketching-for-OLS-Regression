import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_california_housing
from scipy.linalg import hadamard
import time


def OLS_tm(X,Y,model):
    start = time.perf_counter()
    model.fit(X, Y)
    end = time.perf_counter()
    return end-start


def OLSs_tm(X,Y,k,m,model):
    tm = []
    for i in range(k):
        start = time.perf_counter()
        S = np.random.normal(loc=0, scale=np.sqrt(1/m), size=(m,n))
        model.fit(S@X,S@Y)
        end = time.perf_counter()
        tm.append(end-start)
    return np.mean(tm)


def OLSps_tm(X,Y,k,m,model):
    n,d = X.shape
    p = int(n/k)
    tm = []
    for i in range(k):
        idx_head = i*p
        idx_tail = (i+1)*p
        # partition data set 
        Xi = X[idx_head:idx_tail,:]
        Yi = Y[idx_head:idx_tail]
        # truncate sketching matrix
        start = time.perf_counter()
        S = np.random.normal(loc=0, scale=np.sqrt(1/m), size=(m,p))
        model.fit(S@Xi,S@Yi)
        end = time.perf_counter()
        tm.append(end-start)
    return np.mean(tm)


def OLShs_tm(X,Y,k,m,S,model):
    n0,d = X.shape
    rng = np.random.default_rng(0)
    tm = []
    H = hadamard(n) / np.sqrt(n)
    for t in range(k):
        signs = rng.choice([-1, 1], size=n)
        indices = rng.choice(n, size=m, replace=False)
        start = time.perf_counter()
        HD = H * signs
        W = np.sqrt(n / m) * HD[indices, :]
        W = W[:,:n0]        
        model.fit(W@X, W@Y)
        end = time.perf_counter()
        tm.append(end-start)
    return np.mean(tm)

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
k_vec = [1,3,5,7,9] 
n_trial = 100

model = LinearRegression(fit_intercept=False)

# ---------------------------------------
# time of standard OLS without sketching  
# ---------------------------------------
tm0_trial = [] 
for t in range(n_trial):
    tm0_trial.append(OLS_tm(X,Y,model))
tm0_k = np.full(len(k_vec), np.mean(tm0_trial))

# ---------------------------------------------------
# time of distributed sketching based OLS per worker
# ---------------------------------------------------
tm1_k = [] # existing approach 
tm2_k = [] # proposed approach 
for k in k_vec:
    tm1_trial = []
    tm2_trial = []
    for t in range(n_trial):
        print([k,t])
        tm1_trial.append(OLSs_tm(X,Y,k,m,model))
        tm2_trial.append(OLSps_tm(X,Y,k,m,model))
    tm1_k.append(np.mean(tm1_trial))
    tm2_k.append(np.mean(tm2_trial))

# ----------------
# Plot Time Curve 
# ----------------
marksize = 10
fontsize = 14
plt.figure()
plt1, = plt.plot(k_vec,tm1_k, label='$\\bar{\\beta}$', marker='s',markersize=marksize, markerfacecolor="None")
plt2, = plt.plot(k_vec,tm2_k, label='$\\bar{\\theta}$', marker='d',markersize=marksize, markerfacecolor="None")
plt0, = plt.plot(k_vec,tm0_k, label='OLS', linestyle='--', marker='.', markersize=marksize, markerfacecolor="None")
plt.xlabel('k')
plt.ylabel('Model Training Time per Worker')
plt.xticks(k_vec)
plt.legend(loc='upper right',fontsize=fontsize)
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3,fontsize=fontsize)
plt.show()





