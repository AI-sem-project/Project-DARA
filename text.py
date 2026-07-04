import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel("Datasets for DARA")
y = data.iloc[:, -1].values
features=data.columns[:-1]  # all columns except last
for feature in features:
    x=data[[feature]].values
    # splitting data into training and testing data sets
    np.random.seed(42)
    indices=np.random.permutation(len(x))
    train_size=int(0.8 * len(x))
    train_indices=indices[:train_size]
    test_indices=indices[train_size:]
    x_train=x[train_indices]
    y_train=y[train_indices]
    x_test=x[test_indices]
    y_test=y[test_indices]
    # calculating mean values
    mean_x=np.mean(x_test)
    mean_y=np.mean(y_test)
    # calculate slope(m)
    numerator=np.sum((x_train - mean_x) * (y_train - mean_y))
    denominator=np.sum((x_train - mean_x) ** 2)
    slope=numerator / denominator
    # calculate intercept
    intercept=mean_y - slope * mean_x
    # predict house price
    y_pred=(slope * x_test) + intercept
    # calculate Error - R^2 needs the TEST set own mean as baseline
    mean_y_test=np.mean(y_test)
    mse=np.mean((y_test - y_pred) ** 2)  # mean squared error
    mae=np.mean(np.abs(y_test - y_pred))  # mean absolute error
    ss_residual=np.sum((y_test - y_pred) ** 2)
    ss_total=np.sum((y_test - mean_y_test) ** 2)
    r_squared=1 - (ss_residual / ss_total)  # r2 score
    print("Feature =", x)
    print(f"Slope={slope:.4f}")
    print(f"Intercept={intercept:.4f}")
    print(f"MSE={float(mse):.4f}")
    print(f"MAE={float(mae):.4f}")
    print(f"R^2={float(r_squared):.4f}")
    # plot Graph
    plt.figure(figsize=(10.0, 10.0))
    plt.scatter(x_test, y_test, color="red", label="Test Data")
    plt.plot(x_test, y_pred, color="blue", label="Regression line, linewidth =2")
    plt.title("simple linear regression")
    plt.xlabel(f"feature {feature + 1}")
    plt.ylabel("house price")
    plt.legend()
    plt.grid(True)
    plt.show()
