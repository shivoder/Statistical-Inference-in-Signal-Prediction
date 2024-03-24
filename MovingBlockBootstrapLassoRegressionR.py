# Moving block bootstrapping lasso regression

import numpy as np
import yfinance as yf
from sklearn.linear_model import Lasso
import random
import matplotlib.pyplot as plt

# Fetching stock price data for Reliance from Yahoo Finance
reliance = yf.download('RELIANCE.NS', period='100d')
stock_prices = reliance['Close'].values  # Using 'Close' prices as stock prices

# Create sliding windows and corresponding labels
X = []
Y = []
window_size = 10
for i in range(len(stock_prices) - window_size):
    X.append(stock_prices[i:i+window_size])  # Input features
    Y.append(stock_prices[i+window_size])    # Output label

X_train = X[0:54]
Y_train = Y[0:54]
X_test = X[54:90]
Y_test = Y[54:90]

X = np.array(X)
Y = np.array(Y)

# Number of bootstrap samples
num_bootstrap = 54
# Bootstrap resampling
bootstrap_X = []
bootstrap_Y = []
for _ in range(num_bootstrap):
    indices = random.randint(0, 53)
    bootstrap_X.append(X_train[indices])
    bootstrap_Y.append(Y_train[indices])

# Convert lists to numpy arrays
bootstrap_X = np.array(bootstrap_X)
bootstrap_Y = np.array(bootstrap_Y)

# Fit data into Lasso regression model
alpha = 0.01  # Regularization strength
model = Lasso(alpha=alpha)
model.fit(bootstrap_X, bootstrap_Y)

# Print the coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

coefficients = np.array(model.coef_)
Y_predicted = []
x = X_train[-1]
for i in range(36):
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
