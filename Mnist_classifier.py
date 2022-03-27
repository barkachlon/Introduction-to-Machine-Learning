#Bar Kachlon 207630864
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

mnist = fetch_openml('mnist_784')
X = mnist['data'].astype('float64')
t = mnist['target']
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
t = t[permutation]
X = X.reshape((X.shape[0], -1)) #This line flattens the image into a vector of size 784
X_train, X_test, t_train, t_test = train_test_split(X, t, train_size=0.6)#60% of the data goes to train set.
X_test, X_valid, t_test, t_valid = train_test_split(X_test, t_test, test_size=0.5)#half goes to validation set and half go to test set.

##create ones vector and add them to the end of the sets.
ones_vec_train = np.ones((X_train.shape[0], 1))
ones_vec_valid = np.ones((X_valid.shape[0], 1))
ones_vec_test = np.ones((X_test.shape[0], 1))
X_train = np.hstack((X_train, ones_vec_train))
X_valid = np.hstack((X_valid, ones_vec_valid))
X_test = np.hstack((X_test, ones_vec_test))
# The next lines standardize the images
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)
#create weight vectors
Weight_mat = np.random.rand(10, 785)*(10**(-3))
#create one-hot t vectors
t_train_one_hot = np.zeros((t_train.shape[0], 10))
t_valid_one_hot = np.zeros((t_valid.shape[0], 10))
t_test_one_hot = np.zeros((t_test.shape[0], 10))
for i in range(t_train.shape[0]):
    t_train_one_hot[i][int(t_train[i])] = 1
for i in range(t_valid.shape[0]):
    t_valid_one_hot[i][int(t_valid[i])] = 1
    t_test_one_hot[i][int(t_test[i])] = 1
##calculate y_n_k
nom_train = np.exp(np.matmul(Weight_mat, X_train.T))
nom_valid = np.exp(np.matmul(Weight_mat, X_valid.T))
y_train = (nom_train/nom_train.sum(axis=0, dtype=float)).T
y_valid = (nom_valid/nom_valid.sum(axis=0, dtype=float)).T

##cross entropy loss
E = [-1*(np.multiply(t_train_one_hot, (np.log(y_train)))).sum()]
#set variables
accuracy = 0
eta = 0.0001
counter = 0
iter = 0
acc_list = []
##Check first iteration's accuarcy
for i in range(t_valid.shape[0]):
    if np.argmax(y_valid[i]) == int(t_valid[i]):
        counter += 1
accuracy = (counter/len(t_valid))*100
acc_list.append(accuracy)
print("first iteration's accuracy:", accuracy)
##minimze E by using GD
grad_E = (X_train.T.dot(y_train-t_train_one_hot)).T
while eta > 0.0000001:
    counter = 0
    Weight_mat -= eta * grad_E
    nom_train = np.exp(np.matmul(Weight_mat, X_train.T))
    nom_valid = np.exp(np.matmul(Weight_mat, X_valid.T))
    y_train = (nom_train / nom_train.sum(axis=0)).T
    y_valid = (nom_valid / nom_valid.sum(axis=0)).T
    grad_E = (X_train.T.dot(y_train-t_train_one_hot)).T
    for i in range(t_valid.shape[0]):
        if np.argmax(y_valid[i]) == int(t_valid[i]):
            counter += 1
    accuracy = (counter / len(t_valid)) * 100
    eta = eta/2
    print("next iteration's accuracy:", accuracy)
    temp_E = (-1)*(np.multiply(t_train_one_hot, (np.log(y_train)))).sum()
    E.append(temp_E)
    acc_list.append(accuracy)
    iter += 1
#plt cross entropy loss and accuracy as a function of iteration
plt.figure()
plt.subplot(211)
plt.plot(range(iter+1), E)
plt.title('cross entropy loss as a function of iteration')
plt.ylabel('Cross Entropy Loss Function')
plt.subplot(212)
plt.plot(range(iter+1), acc_list)
plt.title('validation set accuracy as a function of iteration')
plt.xlabel('iteration')
plt.ylabel('Accuracy')
plt.show()

#calculate last iteration accuracy for each set
counter = 0
for i in range(t_train.shape[0]):
    if np.argmax(y_train[i]) == int(t_train[i]):
        counter += 1
train_accuracy = (counter / len(t_train)) * 100
nom_test = np.exp(np.matmul(Weight_mat, X_test.T))
y_test = (nom_test / nom_test.sum(axis=0)).T
counter = 0
for i in range(t_test.shape[0]):
    if np.argmax(y_test[i]) == int(t_test[i]):
        counter += 1
test_accuracy = (counter / len(t_test)) * 100
print("Train accuracy:", train_accuracy, "%")
print("Valid accuracy:", accuracy, "%")
print("Test accuracy:", test_accuracy, "%")
