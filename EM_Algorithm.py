#Bar kachlon 207630864
#sector A
import numpy as np
import matplotlib.pyplot as plt

#Define Variables
Iter = 100
N = 2000
M = 2
c_1 = 0.5
c_2 = 0.5
binomial_samples = np.random.binomial(1, 0.5, size=(N,1))
not_binomial_samples = np.logical_not(binomial_samples)
mu_1 = np.array([1, 1])
mu_2 = np.array([3, 3])
Sigma_1 = np.array([[1, 0], [0, 2]])
Sigma_2 = np.array([[2, 0], [0, 0.5]])
X_1_rand = np.random.multivariate_normal(mu_1, Sigma_1, N)
X_2_rand = np.random.multivariate_normal(mu_2, Sigma_2, N)
X = X_1_rand*binomial_samples+X_2_rand*not_binomial_samples
print("Real parameters: c_1=", c_1, "mu_1=", mu_1, "sigma_1=", Sigma_1, "c_2=", c_2,"mu_2=", mu_2, "sigma_2=", Sigma_2)
#random variables
c_1_hat = np.random.rand()
c_2_hat = 1-c_1_hat
mu_1_hat = np.random.uniform(0, 5, size=(1, 2))
mu_2_hat = np.random.uniform(0, 5, size=(1, 2))
Sigma_1_hat = np.random.uniform(0, 5, size=(2, 2))
Sigma_2_hat = np.random.uniform(0, 5, size=(2, 2))
Sigma_1_hat[0, 1] = 0
Sigma_1_hat[1, 0] = 0
Sigma_2_hat[0, 1] = 0
Sigma_2_hat[1, 0] = 0
#_________________________________________________________________________________#

while Iter > 0:
    X_given_l1 = 1 / ((2 * np.pi) * (abs(np.linalg.det(Sigma_1_hat)) ** 0.5)) * np.diag(np.exp(-0.5 * np.matmul((X -mu_1_hat), np.matmul(np.linalg.inv(Sigma_1_hat), (X - mu_1_hat).T))))
    X_given_l2 = 1 / ((2 * np.pi) * (abs(np.linalg.det(Sigma_2_hat)) ** 0.5)) * np.diag(np.exp(-0.5 * np.matmul((X -mu_2_hat), np.matmul(np.linalg.inv(Sigma_2_hat), (X - mu_2_hat).T))))
    alpha_l1 = (c_1_hat * X_given_l1) / ((X_given_l1 * c_1_hat) + (X_given_l2 * c_2_hat))
    alpha_l2 = (c_2_hat * X_given_l2) / ((X_given_l1 * c_1_hat) + (X_given_l2 * c_2_hat))
    c_1_hat = alpha_l1.sum(axis=0) / N
    c_2_hat = alpha_l2.sum(axis=0) / N
    mu_1_hat = np.matmul(alpha_l1.T, X) / np.sum(alpha_l1, axis=0)
    mu_2_hat = np.matmul(alpha_l2.T, X) / np.sum(alpha_l2, axis=0)
    Sigma_1_hat, Sigma_2_hat = Sigma_1_hat, Sigma_2_hat
    Sigma_1_hat[0, 0] = np.matmul(alpha_l1.T, ((X[:, 0] - mu_1_hat[0]) ** 2)) / np.sum(alpha_l1, axis=0)
    Sigma_1_hat[1, 1] = np.matmul(alpha_l1.T, ((X[:, 1] - mu_1_hat[1]) ** 2)) / np.sum(alpha_l1, axis=0)
    Sigma_2_hat[0, 0] = np.matmul(alpha_l2.T, ((X[:, 0] - mu_2_hat[0]) ** 2)) / np.sum(alpha_l2, axis=0)
    Sigma_2_hat[1, 1] = np.matmul(alpha_l2.T, ((X[:, 1] - mu_2_hat[1]) ** 2))/ np.sum(alpha_l2, axis=0)
    Iter = Iter-1
    if Iter == 98:
        print("Parameters after 2 iteration: c_1=", c_1_hat, "mu_1=", mu_1_hat, "sigma_1=", Sigma_1_hat, "c_2=", c_2_hat, "mu_2=", mu_2_hat, "sigma_2=", Sigma_2_hat )
        print(c_2_hat, mu_2_hat, Sigma_2_hat)
    if Iter == 90:
        print("Parameters after 10 iteration: c_1=", c_1_hat, "mu_1=", mu_1_hat, "sigma_1=", Sigma_1_hat, "c_2=", c_2_hat, "mu_2=", mu_2_hat, "sigma_2=", Sigma_2_hat )
print("Parameters after 100 iteration: c_1=", c_1_hat, "mu_1=", mu_1_hat, "sigma_1=", Sigma_1_hat, "c_2=", c_2_hat,"mu_2=", mu_2_hat, "sigma_2=", Sigma_2_hat)

#sector B
#Define Variables
Iter = 100
N = 2000
binomial_samples = np.random.binomial(1, 0.5, size=(N, 1))
not_binomial_samples = np.logical_not(binomial_samples)
mu_1 = np.array([1, 1])
mu_2 = np.array([3, 3])
Sigma_1 = np.array([[1, 0], [0, 2]])
Sigma_2 = np.array([[2, 0], [0, 0.5]])
X_1_rand = np.random.multivariate_normal(mu_1, Sigma_1, N)
X_2_rand = np.random.multivariate_normal(mu_2, Sigma_2, N)
X = X_1_rand*binomial_samples+X_2_rand*not_binomial_samples
z = np.random.uniform(0, 5, size=(2, 2))
w = np.zeros((N,))
while Iter > 0:
    #Cluster dataset vectors using nearest neighbor rule
    for i in range(N):
        d_1 = (X[i, 0] - z[0, 0])**2 + (X[i, 1] - z[0, 1])**2
        d_2 = (X[i, 0] - z[1, 0])**2 + (X[i, 1] - z[1, 1])**2
        if d_1 < d_2:
            w[i] = 1
        else:
            w[i] = 0
    #Compute new representatives to minimize D
    w_i = np.row_stack([w, w])
    z = np.row_stack([np.sum(w_i.T*X, axis=0)/np.sum(w_i.T, axis=0), np.sum(np.logical_not(w_i).T*X, axis=0)/np.sum(np.logical_not(w_i).T, axis=0)])
    Iter = Iter - 1
    if Iter == 98:
        print("Mean vector after 2 iteration: ", z)
    if Iter == 90:
        print("Mean vector after 10 iteration: ", z)
print("Mean vector after 100 iteration: ", z)