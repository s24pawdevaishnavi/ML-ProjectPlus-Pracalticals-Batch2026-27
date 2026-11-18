# Practical No. 3
# Title: Implement simple/multiple linear regression using numpy and using scikit-learn & residuals, RMSE, R2 value

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data
X = np.array([10, 22, 43, 94, 50]).reshape(-1, 1)
y = np.array([22, 14, 45, 94, 46])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
print("R2 Score:", r2_score(y, y_pred))

# Plot regression line
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()
plt.show()


"""
=================== OUTPUT (text output; plots open in a window when run normally) ===================
Slope: 0.9383259911894268
Intercept: 3.101321585903108
RMSE: 6.5009657025767575
R2 Score: 0.9456611871707477
"""
