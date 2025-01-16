import numpy as np
import pandas as pd
#use pandas to load real_estate_dataset.csv
df = pd.read_csv('real_estate_dataset.csv')

#get the number of samples and features
nsamples, nfeatures = df.shape
print(f"Number of samples , features: {nsamples}, {nfeatures}")

#get the names of the columns
columns = df.columns

#save the columns names in a file named columns.csv
np.savetxt('columns.csv', columns, fmt='%s')

# use square_feet, Garage_size, location_score, Distance to centre as features
X = df[['Square_Feet', 'Garage_Size', 'Location_Score', 'Distance_to_Center']]

#use price as target column
y = df['Price'].values

print(f"Shape of x: {X.shape}")
print(f"Type of x: {type(X)}")

#get the number of samples and features in X
n_samples, n_features = X.shape

#Build a linear model to predict the price from the four features in X
# make an array of coeffs of the size of n_features+1. iniatialise to 1.
coeffs = np.ones(n_features+1)

# predict the price for each sample in X
predictions_bydefn = X @ coeffs[1:] + coeffs[0]

#append a column of ones to X so that bias can be included in the coeffs
X = np.hstack((np.ones((n_samples, 1)), X))

#predict the price for each sample in X
predictions = X @ coeffs

#see if all the entries in predictions_bydefn and predictions are the same
is_same = np.allclose(predictions_bydefn, predictions)

print(f"Are the predictions the same? {is_same}")

#calculate the error using predictions and y
errors = predictions - y

#calculate the relative error
relative_error = errors / y

#calculate the mean square of errors using a loop
loss_loop = 0
for i in range(n_samples):
    loss_loop += errors[i] ** 2

# calculate the mean of square of errors using matrix operations
loss_matrix = np.transpose(errors) @ errors / n_samples

#compare the two methods of calculating the mean square error
is_diff = np.abs(loss_loop - loss_matrix)
print(f"are the two methods different? {is_diff}")

#print the size of errors and its L2 norm
print(f"Size of errors: {errors.shape}")
print(f"L2 norm of errors: {np.linalg.norm(errors)}")

#calculate the relative error and L2 norm of the relative error
print(f"Relative error: {np.linalg.norm(errors/y)}")

# what is my optimization problem?
# I want to find the coeffs that minimize the mean square error
# this problem is called a least squares problem
#objective function : f(coeffs)  = 1/n_samples * \sum_{i=1}^{n_samples} (y_i - coeffs[0] - coeffs[1] * x_i[1] - coeffs[2] * x_i[2] - coeffs[3] * x_i[3] - coeffs[4] * x_i[4])^2

#How do I find a solution?
# A solution can be found by setting the gradient of the objective function to zero
# The gradient is a vector of partial derivatives of the objective function with respect to each of the coefficients

#solution lies in no of features plus 1 dimensional space
#can find a solution by searching for the coefficients at which the gradient of the objective function is zero
#or directly substitute the gradient of the objective function to zero and solve for the coefficients

#write the loss_matrix in terms of the data and coeffs
loss_matrix = (y- X @ coeffs).T @ (y - X @ coeffs) / n_samples

#calculate the gradient of the loss_matrix with respect to the coeffs
gradient = -2/n_samples * X.T @ (y - X @ coeffs)

#set the grad_matrix to zero and solve for the coeffs
#X.T @ y = X.T @ X @ coeffs
# X.T @ X @ coeffs = X.T @ y(normal equation)
# coeffs = (X.T @ X)^-1 @ X.T @ y

coeffs = np.linalg.inv(X.T @ X) @ X.T @ y

#save the coeffs in a file named coeffs.txt
np.savetxt('coeffs.csv', coeffs)

#calculate the predictions using the optimal coeffs
predictions_model = X @ coeffs

#calculate the errors using the optimal coeffs
errors_model = predictions_model - y

#print the L2 norm of the errors
print(f"L2 norm of errors(normal): {np.linalg.norm(errors_model)}")

#print the L2 norm of the relative errors_model
print(f"Relative error: {np.linalg.norm(errors_model/y)}")

#use all the features in the dataset to build a linear model
X = df.drop('Price', axis=1).values
y = df['Price'].values

#get the number of samples and features in X
n_samples, n_features = X.shape
print(f"Number of samples, features: {n_samples}, {n_features}")

#solve the linear model using the normal equation
X = np.hstack((np.ones((n_samples, 1)), X))
coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
#save the coeffs in a file named coeffs_all.csv
np.savetxt('coeffs_all.csv', coeffs, delimiter=',')

#calculate the rank of X.T @ X
rank = np.linalg.matrix_rank(X.T @ X)

#solve the normal equation using matrix decomposition when X.T @ X is not invertible
#QR factorization
Q, R = np.linalg.qr(X)

print(f"Shape of Q: {Q.shape}")
print(f"Shape of R: {R.shape}")

#write R to a file name in R.csv
np.savetxt('R.csv', R, delimiter=',')


#R*coeffs = b
#X.T @X = R.T @ Q.T @ Q @ R = R.T @ R
# X.T @ y = R.T @ Q.T @ y
# R*coeffs = Q.T @ y

b= Q.T @ y
coeffs_qr = np.linalg.inv(R) @ b
#loop to solve R*coeffs = b using back substitution
coeffs_qr_loop = np.zeros(n_features+1)
for i in range(n_features, -1, -1):
    coeffs_qr_loop[i] = b[i]
    for j in range(i+1, n_features+1):
        coeffs_qr_loop[i] -= R[i, j] * coeffs_qr_loop[j]
    coeffs_qr_loop[i] /= R[i, i]

#solve the normal equation using SVD decomposition
#X = U @ S @ V.T
#calculate the coeffs using the SVD decomposition
U, S, Vt = np.linalg.svd(X, full_matrices = False)
S_inv = np.diag(1/S)
coeffs_svd = Vt.T @ S_inv @ U.T @ y

#save the coeffs in a file named coeffs_svd.csv
np.savetxt('coeffs_svd.csv', coeffs_svd, delimiter=',')

#calculate the predictions using the optimal coeffs
predictions_svd = X @ coeffs_svd

#calculate the errors using the optimal coeffs
errors_svd = y - predictions_svd

#calculate the L2 norm of the errors
l2_norm_errors_svd = np.linalg.norm(errors_svd)
print(f"L2 norm of errors(SVD): {l2_norm_errors_svd}")

#calculate the relative error
relative_error_svd = errors_svd / y

#save the predictions in a file named predictions_svd.csv
np.savetxt('predictions_svd.csv', predictions_svd, delimiter=',')
print(f"Relative error(SVD):", np.linalg.norm(relative_error_svd))


#eigen value decomposition of a square matrix
# A = V @ D @ V.T
# A^-1 = V @ D^-1 @ V.T
# X*coeffs = y
# A = X.T @ X

U, S, Vt = np.linalg.svd(X, full_matrices=False)
#find the inverse of X in the least square sense is pseudo inverse
#Normal equation: X.T @ X @ coeffs = X.T @ y
#Xdagger = (X.T @ X)^-1 @ X.T
#verifying if Q is orthogonal
sol = Q.T @ Q
np.savetxt('sol.csv', sol, delimiter=',')









