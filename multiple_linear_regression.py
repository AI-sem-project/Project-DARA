import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# load Dataset
data=pd.read_excel("Datasets for DARA.xlsx")
# first 6 columns are house features
x=data.iloc[:, :-1].values
# last column is price of houses
y=data.iloc[:, -1].values
# splitting data into training and testing data
np.random.seed(42)
indices=np.random.permutation(len(x))
train_size=int(0.8*len(x))
train_indices=indices[:train_size]
test_indices=indices[train_size:]
x_train=x[train_indices]
y_train=y[train_indices]
x_test=x[test_indices]
y_test=y[test_indices]
# Add Bias column
ones_train=np.ones((x_train.shape[0],1))
ones_test=np.ones((x_test.shape[0],1))
x_train=np.hstack((ones_train,x_train))
x_test=np.hstack((ones_test,x_test))
# Train model normal equation
beta=np.linalg.inv(x_train.T @ x_train @ x_train.T @ y_train)
# Prediction of house price
y_pred=x_test @ beta
# Calculating Error
mae=np.mean(np.abs(y_test - y_pred))  # Mean Absolute Error
mse=np.mean((y_test-y_pred) ** 2)  # Mean squared Error
ss_residual=np.sum((y_test-y_pred) ** 2)
ss_total=np.sum((y_test - np.mean(y_test)) ** 2)
r_squared=1 - (ss_residual / ss_total)
# print results
print("\nRegression coefficients")
print(beta)
print("\nModel performance")
print(f"MAE: {float(mae):.4f}")
print(f"MSE: {float(mse):.4f}")
print(f"R-squared: {float(r_squared):.4f}")
# plot Actual vs predicted plot
plt.figure(figsize=(10.0,10.0))
plt.scatter(y_test, y_pred, color='blue')
minimum=min(y_test.min(), y_pred.min())
maximum=max(y_test.max(), y_pred.max())
plt.plot([maximum, minimum], [minimum, maximum], color='red', linewidth=2)
plt.title("Multiple Linear Regression")
plt.xlabel("Actual Values")
plt.ylabel("Predicted values")
plt.grid(True)
plt.show()

