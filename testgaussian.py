import matplotlib.pyplot as plt
import numpy as np


def RBF_kernel(xn, xm, l=1):
    """
    Inputs:
        xn: row n of x
        xm: row m of x
        l:  kernel hyperparameter, set to 1 by default
    Outputs:
        K:  kernel matrix element: K[n, m] = k(xn, xm)
    """
    K = np.exp(-np.linalg.norm(xn - xm)**2 / (2 * l**2))
    return K


def make_RBF_kernel(X, l=1, sigma=0):
    """
    Inputs:
        X: set of φ rows of inputs
        l: kernel hyperparameter, set to 1 by default
        sigma: Gaussian noise std dev, set to 0 by default
    Outputs:
        K:  Covariance matrix 
    """
    K = np.zeros([len(X), len(X)])
    for i in range(len(X)):
        for j in range(len(X)):
            K[i, j] = RBF_kernel(X[i], X[j], l)
    return K + sigma * np.eye(len(K))


def gaussian_process_predict_mean(X, y, X_new):
    """
    Inputs:
        X: set of φ rows of inputs
        y: set of φ observations 
        X_new: new input 
    Outputs:
        y_new: predicted target corresponding to X_new
    """
    rbf_kernel = make_RBF_kernel(np.vstack([X, X_new]))
    K = rbf_kernel[:len(X), :len(X)]
    k = rbf_kernel[:len(X), -1]
    return np.dot(np.dot(k, np.linalg.inv(K)), y)


def gaussian_process_predict_std(X, X_new):
    """
    Inputs:
        X: set of φ rows of inputs
        X_new: new input
    Outputs:
        y_std: std dev. corresponding to X_new
    """
    rbf_kernel = make_RBF_kernel(np.vstack([X, X_new]))
    K = rbf_kernel[:len(X), :len(X)]
    k = rbf_kernel[:len(X), -1]
    return rbf_kernel[-1, -1] - np.dot(np.dot(k, np.linalg.inv(K)), k)

def f(x):
    return (x-5) ** 2
# Training data x and y:
X = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
print(X.shape)
X = np.random.rand(15) * 10
print(X.shape)
y = f(X)
X = X.reshape(-1, 1)
# New input to predict:
X_new = np.array([5.5])
# Calculate and print the new predicted value of y:
mean_pred = gaussian_process_predict_mean(X, y, X_new)
print("mean predict :{}".format(mean_pred))
# Calculate and print the corresponding standard deviation:
sigma_pred = np.sqrt(gaussian_process_predict_std(X, X_new))
print("std predict :{}".format(sigma_pred))

# Range of x to obtain the confidence intervals.
x = np.linspace(0, 10, 1000)
# Obtain the corresponding mean and standard deviations.
y_pred = []
y_std = []
for i in range(len(x)):
    X_new = np.array([x[i]])
    y_pred.append(gaussian_process_predict_mean(X, y, X_new))
    y_std.append(np.sqrt(gaussian_process_predict_std(X, X_new)))

y_pred = np.array(y_pred)
y_std = np.array(y_std)
plt.figure(figsize=(15, 5))
plt.plot(x, f(x), "r")
plt.plot(X, y, "ro")
plt.plot(x, y_pred, "b-")
plt.fill(np.hstack([x, x[::-1]]),
         np.hstack([y_pred - 1.9600 * y_std,
                   (y_pred + 1.9600 * y_std)[::-1]]),
         alpha=0.5, fc="b")
plt.xlabel("$x$", fontsize=14)
plt.ylabel("$f(x)$", fontsize=14)
plt.legend(["$y = x^2$", "Observations", "Predictions",
           "95% Confidence Interval"], fontsize=14)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
