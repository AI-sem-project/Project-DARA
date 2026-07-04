import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv("Datasets for DARA.xlsx")

# Features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

