# local block bootstrap ridge regression

import numpy as np
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import random
import matplotlib.pyplot as plt

# Fetching stock price data for Reliance from Yahoo Finance
reliance = yf.download('RELIANCE.NS', period='100d')
stock_prices = reliance['Close'].values  # Using 'Close' prices as stock prices

# Create sliding windows and corresponding labels
X = []
Y = []
window_size = 10

for i in range(9):
    X.append(stock_prices[i*10:i*10+window_size])  # Input features
    Y.append(stock_prices[i*10+window_size])    # Output label

X_train = X[0:6]
Y_train = Y[0:6]
X_test = stock_prices[60:]
Y_test = stock_prices[60:]

X = np.array(X)
Y = np.array(Y)

e = random.randint(0, 2)
bootstrap_X = []
bootstrap_Y = []
bootstrap_X.append(stock_prices[e:e+10])
bootstrap_Y.append(stock_prices[e+10])

for i in range(1, 6):
    e = random.randint(-3, 3)
    bootstrap_X.append(stock_prices[i*10+e:i*10+e+10])
    bootstrap_Y.append(stock_prices[i*10+e+10])

num_bootstrap = 6
# Convert lists to numpy arrays
bootstrap_X = np.array(bootstrap_X)
bootstrap_Y = np.array(bootstrap_Y)

# Define a range of alpha values to search
alphas = np.logspace(-4, 4, 9)

# Perform grid search to find the optimal alpha
ridge = Ridge()
grid_search = GridSearchCV(estimator=ridge, param_grid=dict(alpha=alphas), cv=5)
grid_search.fit(bootstrap_X, bootstrap_Y)

# Get the best alpha value
best_alpha = grid_search.best_estimator_.alpha

# Fit data into Ridge regression model with the optimal alpha
model = Ridge(alpha=best_alpha)
model.fit(bootstrap_X, bootstrap_Y)

# Print the coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

coefficients = np.array(model.coef_)
Y_predicted = []
x = X_train[-1]
for i in range(40):
    y = np.dot(x, coefficients) + model.intercept_
    Y_predicted.append(y)
    x = x[1:]
    x = np.append(x, y)

Y_test = np.array(Y_test)
Y_predicted = np.array(Y_predicted)

error = ((abs(Y_test - Y_predicted))/Y_test)*100
print("Error:", error)

Y_predic = []
x = stock_prices[90:]
for i in range(100):
    y = np.dot(x, coefficients) + model.intercept_
    Y_predic.append(y)
    x = x[1:]
    x = np.append(x, y)

print("Predicted Values:", Y_predic)

# Plotting
plt.figure(figsize=(10, 5))

# Plotting Error
plt.plot(error, color='red')
plt.title('Prediction Error')
plt.xlabel('Days')
plt.ylabel('Error')

# Plotting Predicted Values
plt.figure(figsize=(10, 5))
plt.plot(range(101, 201), Y_predic, color='blue')
plt.title('Predicted Stock Prices')
plt.xlabel('Days')
plt.ylabel('Price')

plt.show()
