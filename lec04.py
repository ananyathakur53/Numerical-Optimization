# iterative optimization methods
#line search methods and trust region methods
import numpy as np
import matplotlib.pyplot as plt

#create a 1D scalar function
def f(x):
    return 2*x**2 + 3*x + 4

#visualise this function for x in range -10 to 10

x = np.linspace(-10, 10, 100)
y = f(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('A quadratic function')
plt.close()
#defining the gradient of the function
def grad_f(x):
    return 4*x + 3

#make an initial guess of x_optimum = 5
x_optimum = 5

#print the value of the function at x_optimum and its gradient
print(f"f(x_optimum) = {f(x_optimum)}") 
print(f"grad_f(x_optimum) = {grad_f(x_optimum)}")

#in a 1D problem, the gradient is a scalar either positive or negative
#in this case the gradient is (4x + 3)i
# if the gradient is positive, the function is increasing
# we are moving in the negative direction of the gradient
direction_of_descent = -grad_f(x)

#let us consider a step_length that defines the amount of movementthat I will perform in the direction of the gradient
#x_new = x_optimum + step_length * direction_of_descent

#what is the value of the function at the new x (x_new)
# f_x_new = f(x_new)= f(x_optimum + step_length * direction_of_descent)
#expand f in terms of its Taylor approximation around x_0

#f(x) = f(x_0) + grad_f(x_0) * step_length * direction_of_descent + 1/2 * f_grad_grad_f^T * step_length^2 * direction_of_descent^2

#making first order approximation of f(x) around x_0
#f(x_new) = f(x_0) + grad_f(x_0) * step_length * direction_of_descent + O(step_length^2)

#we need f(x_new) to be minimum
# we make f(x_new) as the function of the step size
# Goal is to find the step_length that minimizes f(x_new)

#two conditions are used to find the step_length 1. wolfe conditions 2. Armijo conditions
#step length has to be greater than 0

# consider f2(x1, x2) = x1^2 + x2^2 + 2x1x2
def f2(x1, x2):
    return x1**2 + x2**2 + 0.5*x1*x2

#define the gradient of the function2
def grad_f2(x1, x2):
    return np.array([2*x1 + 0.5*x2, 2*x2 + 0.5*x1])


#visualise this function for x1 and x2 in range -10 to 10
x1 = np.linspace(-4, 4, 100)
x2 = np.linspace(-4, 4, 100)
[X1, X2] = np.meshgrid(x1, x2)
Y = f2(X1, X2)

#plt.contour(X1, X2, Y, 20) #20 is the number of contour lines

#use a perceptually uniform colormap
#same color maps are used for the same data
#red is used for the highest value and blue is used for the lowest value
#for diverging maps zero is in the middle

#use blues matplot lib colormap
cfp = plt.contourf(X1, X2, Y, 200)
#set color axis limits to 0 to 10
plt.clim(0, 10)
plt.colorbar(cfp)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('A quadratic function of two variables')
plt.savefig('quadratic_function.png')
# plt.show()


#for higher dimension problems we project the problem in 2D and then visualise  it
# function to find the minimum 
dir_of_descent = -grad_f2(x1, x2) #steepest descent direction
#steps
#1.function to find the minimum of a 1D scalar function
#2. use that together with the gradient of the 2D function to code the iterative gradient descent
#second order approximation of f(x) around x_0
#f_x_new(step_length) = f(x_0) + grad_f(x_0) * step_length * direction_of_descent + 1/2 * f_grad_grad_f^T * step_length^2 * direction_of_descent^2 

def backtracking_line_search(f, grad_f, x, dir_descent, alpha=1.0, rho=0.8, c1=1e-4):
    while f(*(x + alpha * dir_descent)) > f(*x) + c1 * alpha * grad_f(*x).dot(dir_descent):
        alpha *= rho
    return alpha

# Gradient Descent Implementation
def gradient_descent(f, grad_f, x0, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad = grad_f(*x)  # Gradient at current point
        dir_descent = -grad  # Steepest descent direction
        
        # Find optimal step length using backtracking line search
        step_length = backtracking_line_search(f, grad_f, x, dir_descent)
        
        # Update x using the step length
        x_new = x + step_length * dir_descent
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged in {i+1} iterations.")
            return x_new
        x = x_new  # Update current position
    
    print("Maximum iterations reached.")
    return x

# Initial guess
x0 = [4.0, 4.0]

# Perform gradient descent
minimum = gradient_descent(f2, grad_f2, x0)

# Print results
print("Minimum found at:", minimum)
print("Minimum value:", f2(*minimum))


 



