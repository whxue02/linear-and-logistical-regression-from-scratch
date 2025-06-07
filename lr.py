import pandas as pd
import numpy as np

# load in data
teams = pd.read_csv('teams.csv')
print(teams)

# set up equations
X = teams[["athletes", "prev_medals"]].copy()
y = teams[["medals"]].copy()
# add a constant 1 for the intercept
X["intercept"] = 1
# rearrange columns to have intercept first
X = X[["intercept", "athletes", "prev_medals"]]
print(X)
print(y)
# X transpose matrix
X_T = X.T

# final formula: B = (X_TX)-1X_Ty (final formula)
# calculate B using the equation
B = np.linalg.inv(X_T @ X) @ X_T @ y
B.index = X.columns
print(B) 
# 0 = -1.961889 (y-intercept)
# 1 = 0.071112 (athletes coefficient, for every 1 additional athlete, medals increase by 0.071112)
# 2 = 0.734137 (prev_medals coefficient, ^)

# formula for prediction: X @ B
predictions = X @ B
print(predictions)

# SSR (sum of squared residuals) = ((y - predictions) ** 2).sum()
SSR = ((y - predictions) ** 2).sum()
# SST (total sum of squares) = ((y - y.mean()) ** 2).sum()
SST = ((y - y.mean()) ** 2).sum()
# R^2 = 1 - SSR/SST
R_squared = 1 - SSR / SST
print(f"SSR: {SSR}, SST: {SST}, R^2: {R_squared}")

# R^2 = 0.87 
# this means that 87% of the variance in medals can be explained by the number of athletes and previous medals.
 

