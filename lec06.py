import numpy as np
import matplotlib.pyplot as plt
from datamaker import RegressionDatamaker
import warnings
warnings.filterwarnings("ignore")
#create an instance of regression datamaker class
data_maker = RegressionDatamaker(n_samples=100, n_features=1, noise=0.1,seed=42)

#make_data 
X, y, coefs = data_maker.make_data_with_ones()

#save data to a csv file
data_maker.save_data(X, y, "data.csv")

#save coefficients to a csv file
data_maker.save_coefs(coefs, "true_coefs.csv")

#define a linear model
def linear_model(X, theta):
    return X@theta

#make a least squares objective function for regression
def mse_linear_regression(X,y,theta):
    n_samples = X.shape[0]
    err  = linear_model(X, theta) - y
    sum_sq_err = 0
    for i in range(n_samples):
        sum_sq_err += err[i]**2
    return (1/n_samples)*sum_sq_err

#make a function to compute the gradient of the least squares objective function
def grad_mse_linear_regression(X,y,theta):
    n_samples = X.shape[0]
    return (2/n_samples)*X.T@(X@theta-y)

#make a function to perform gradient descent
step_length = 0.1
num_iter = 100
theta0 = np.array([[2],[2]])#initial theta
print(f"number of features: {X.shape[1]}")
print(f"number of samples: {X.shape[0]}")

def gradient_descent(X,y,theta0,mse_linear_regression, grad_mse_linear_regression, step_length, num_iter):
    n_samples, n_features = X.shape
    theta = theta0
    path = theta 
    iter_cnt = 0
    while np.linalg.norm(grad_mse_linear_regression(X,y,theta).any())>1e-6:
        theta = theta - step_length*grad_mse_linear_regression(X,y,theta)
        iter_cnt += 1
        path = np.hstack((path, theta))
        if iter_cnt> num_iter:
            break
        
    return theta, path

#plot the contour of the least squares objective function
def plot_contour(X,y,mse_linear_regression,path,theta,step_length):
    theta0_vals = np.linspace(-5, 10, 100)
    theta1_vals = np.linspace(-5, 10, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            J_vals[i, j] = mse_linear_regression(X, y, np.array([[theta0], [theta1]]))
    plt.contourf(theta0_vals, theta1_vals, J_vals.T, levels=np.arange(0, 100, 10))
    
    #plot the path 
    print(f"Shape of path")
    plt.plot(path[0], path[1], marker='x', color='black')
    # plot the iteration number along the path
    for i in range(path.shape[1]):
        plt.text(path[0,i], path[1,i], str(i), color='black')
    #plot the final theta in red
    plt.plot(theta[0],theta[1], marker='x',color='red')
    plt.xlabel(r'$\theta0$')
    plt.ylabel(r'$\theta1$')
    plt.savefig("contour_mse_with_path_sl{step_length}.png")
    plt.show()
    
    #perform gradient descent
    theta, path = gradient_descent(X,y,theta0,mse_linear_regression,grad_mse_linear_regression,step_length,num_iter)
    print(f"Shape of theta: {theta.shape}")
    #plot the contour of the least squares objective function
    
    
theta, path = gradient_descent(X,y,theta0,mse_linear_regression,grad_mse_linear_regression,step_length,num_iter)
plot_contour(X,y,mse_linear_regression,path,theta, step_length=step_length) 
    
#plot the contour without path
def plot_contour(X,y,mse_linear_regression,id):
    theta0_vals = np.linspace(-5, 5, 100)
    theta1_vals = np.linspace(-5, 5, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            J_vals[i, j] = mse_linear_regression(X, y, np.array([[theta0], [theta1]]))
    plt.contourf(theta0_vals, theta1_vals, J_vals.T, levels=np.arange(0, 100, 10))
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.savefig(f'contour_mse_{id}.png')
    plt.show()
    
plot_contour(X[0:3,:],y[0:3],mse_linear_regression,'0to3')
plot_contour(X[3:6,:],y[3:6],mse_linear_regression,'3to6')

#stochastic gradient descent algorithm
#plots to be made after each iteration to show the path of descent and nature of the contours(one data point at a time)
#when there are numerous peaks and troughs, it behaves better with lesser data points
            