import numpy as np
import matplotlib.pyplot as plt
import random

#create a 2D callable function representing a quadratic function
def f(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0]*x[1]

#create a 2D callable function representing the gradient of the quadratic function
def grad_f(x):
    return np.array([2*x[0] + 0.5*x[1], 2*x[1] + 0.5*x[0]])

#1D backtracking line search method
def backtracking_line_search(f, grad_f, x, direction_of_descent, alpha = 1, beta = 0.8, c= 1e-4):
    while f(x+ alpha*direction_of_descent)> f(x)+c*alpha*grad_f(x).T@direction_of_descent:
        alpha = beta*alpha
    return alpha

#define a method to evaluate if a given direction_of_descent is a valid descent direction
def is_descent_direction (grad_f, x, direction_of_descent):
     return grad_f(x).T@direction_of_descent < 0
    
    
#define a method to perform gradient descent
def gradient_descent(f, grad_f, x0,alpha, beta,  max_iter = 1000, tol=1e-6):
    x = x0
    path = x
    for i in range(max_iter):
        if np.linalg.norm(grad_f(x))<tol:
            break
        direction_of_descent = -grad_f(x)#steepest descent direction
        step_length = backtracking_line_search(f, grad_f, x, direction_of_descent, alpha= alpha, beta= beta)
        x = x + step_length*direction_of_descent
        #save the path of descent in an numpy array
        path = np.vstack((path, x))
    return x, f(x), i, path

#visualise the contour plot of the function f(x)
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])

cfp = plt.contourf(X,Y,Z, levels = np.linspace(0,100,10),cmap = 'Blues',extend='max',vmin=0,vmax=100)
plt.colorbar(cfp)
# plt.savefig("f_contour.png")
# plt.show()


#works cause the callable function is treating the input as a vector
#won't work in c++ and fortran

#set initial guess as x0 = [5, 5]
x0 = np.array([5, 5])

alpha = 1
beta = 0.4
#no of iterations reduce as we increase the value of beta because the step length is reduced
#perform gradient descent
#the function converges to the minima just in one iteration with alpha = 1 and beta = 0.4
x, f_min, num_iter, path = gradient_descent(f, grad_f, x0, alpha, beta)

# print the number of iterations and the final solution
print('Number of iterations:', num_iter)
print('Final solution:', x)

# plot the path of descent on the contour plot
plt.plot(path[:,0], path[:,1], marker = 'o')
# mark the initial guess and the final solution in red and green respectively
plt.plot(x0[0], x0[1], marker = 'o', color = 'red')
plt.plot(x[0], x[1], marker = 'o', color = 'green')
plt.title("Contour plot of the function f(x) and the path of descent")
plt.xlabel("x[0]")
plt.ylabel("x[1]")

#mark every point with iteration number
for i in range(path.shape[0]):
    plt.text(path[i, 0], path[i, 1], str(i), color = 'red')
plt.savefig(f"f_contour_with_descent_alpha{alpha}_beta{beta}.png")
plt.show()
plt.close()

#determine step length and pass it as a hyperparameter to the gradient descent method
#new method for gradient descent
def gradient_descent_nobacktracking(f, grad_f, x0,step_length= 0.1,max_iter = 1000, tol = 1e-6):
    x = x0
    path = x
    for i in range(max_iter):
        if np.linalg.norm(grad_f(x))<tol:
            break
        direction_of_descent = -grad_f(x)#steepest descent direction
        x = x + step_length*direction_of_descent
        #save the path of descent in an numpy array
        path = np.vstack((path, x))
    return x, f(x), i, path

#set initial guess as x0 = [5, 5]
x0 = np.array([5, 5])
step_length = 0.1
#perform gradient descent
x, f_min, num_iter, path = gradient_descent_nobacktracking(f, grad_f, x0, step_length)

#print the number of iterations and the final solution
print('Number of iterations without backtracking:', num_iter)
print('Final solution without backtracking:', x)

# visulaize the contour plot of the function f(x)
cfp = plt.contourf(X,Y,Z, levels = np.linspace(0,100,10),cmap = 'Blues',extend='max',vmin=0,vmax=100)
plt.colorbar(cfp)

# visualize the path of descent on the contour plot
plt.plot(path[:,0], path[:,1], marker = 'o')

#mark the initial guess and final solution in red and green respectively
plt.plot(x0[0], x0[1], marker = 'o', color = 'red', label = 'Initial guess')
plt.plot(x[0], x[1], marker = 'o', color = 'green', label = 'Final solution')
plt.title(f"Contour plot of the function f(x) and the path of descent without backtracking with step_length {step_length}")
plt.xlabel("x[0]")  
plt.ylabel("x[1]")

#mark every point with iteration number
for i in range(path.shape[0]):
    plt.text(path[i, 0], path[i, 1], str(i), color = 'red')
    
plt.savefig(f"f_contour_with_descent_no_backtracking_step_length{step_length}.png")
plt.show()
plt.close()
#homework
#coordinate descent
#hit and trial method of descent direction, check for validity and as soon as we find a valid descent direction, descent.
#visualise the path 
#gradient descent direction is guaranteed to converge as long as we choose the valid direction of descent
def gradient_descent_random(f, grad_f, x0, step_length=0.1,max_iter=1000,tol=1e-6):
    x =x0
    path  = x
    for i in range(max_iter):
        if np.linalg.norm(grad_f(x)) < tol:
            break
        direction_of_descent = np.random.uniform(-1,1,2)
        while not is_descent_direction(grad_f, x, direction_of_descent):
            direction_of_descent = np.random.uniform(-1,1,2)
        x = x + step_length*direction_of_descent
        # save the path of descent in a numpy array
        path = np.vstack((path, x))
    return x, f(x), i, path

# set initial guess as [5,5]
x0 = np.array([5,5])
step_length = 0.1
# perform gradient descent method
x,f_x, num_iter, path = gradient_descent_random(f, grad_f, x0,step_length)

# print the number of iterations and the final solution
print('Number of iterations without backtracking and for random descent direction:', num_iter)
print('Final solution without backtracking and for random descent direction:', x)

# visualize the contour plot of the function f(x)

cfp = plt.contourf(X,Y,Z, levels = np.linspace(0,100,10),cmap = 'Blues',extend='max',vmin=0,vmax=100)
plt.colorbar(cfp)

# visualize the path of descent on the contour plot
plt.plot(path[:,0], path[:,1], marker = 'o')

# mark the initial guess and the final solution in red and green respectively
plt.plot(x0[0], x0[1], marker = 'o', color = 'red')
plt.plot(x[0], x[1], marker = 'o', color = 'green')
plt.title(f"plot of the function f(x) and the path of descent without backtracking and for random descent direction with step_length {step_length}")
plt.xlabel("x[0]")  
plt.ylabel("x[1]")

# mark every point with the iteration number
for i in range(path.shape[0]):
    plt.text(path[i, 0], path[i, 1], str(i), color = 'red')
plt.savefig(f"f_contour_with_descent_no_backtracking_random_descent_step_length{step_length}.png")
plt.show()

    



        
