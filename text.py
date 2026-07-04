import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel("Datasets for DARA")
y = data.iloc[:, -1].values
features = data.columns[:-1]  # all columns except last
for feature in features:
    x = data[[feature]].values
    #splitting data into training and testing data sets
    np.random.seed(42)
    indices = np.random.permutation(len(x))
    train_size = int(0.8 * len(x))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]
    # calculating mean values
    x_mean = np.mean(x_train)
    y_mean = np.mean(y_train)
    # calculate slope(b1)
    b1 = np.sum((x_train - x_mean) * (y_train - y_mean))
    # calculate intercept(b0)
    b0 = y_mean - b1 * x_mean
    # predict house price
    y_pred = b0 + b1 * x_test
    # calculate Error
    MSE = np.mean((y_test - y_pred) ** 2)
    RMSE = np.sqrt(MSE)
    print("Feature=", x)
    print("Slope=", b1)
    print("Intercept=", b0)
    print("MSE=", float(MSE))
    print("RMSE=", float())
    # plot Graph
    plt.scatter(x_test, y_test)
    plt.plot(x_test, y_pred, color="red")
    plt.title("simple linear regression")
    plt.xlabel("Feature")
    plt.ylabel("House price")
    plt.show()
