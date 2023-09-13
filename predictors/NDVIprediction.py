import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Load your CSV data into a Pandas DataFrame
data = pd.read_csv("./data/Potch farm data.csv",sep = ";")

# Split the data into features (X) and target (y)
X = data[[ 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07', 'Month']]
y = data['NDVI']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
linear_model = LinearRegression()
# Train the linear regression model
linear_model.fit(X_train, y_train)


# Make predictions using the linear regression model
linear_predictions = linear_model.predict(X_test)


# Calculate mean, max, and min of predictions
mean_prediction = np.mean(linear_predictions)
max_prediction = np.max(linear_predictions)
min_prediction = np.min(linear_predictions)

print(f"Mean Prediction: {mean_prediction}")
print(f"Max Prediction: {max_prediction}")
print(f"Min Prediction: {min_prediction}")

# Define a test case for linear regression
linear_test_case = np.array([[643.06, 2550.2599999999998, 292.28999999999996, 1959.3499999999997, 3]])

# Make predictions using the linear regression model
linear_prediction = linear_model.predict(linear_test_case)

print(f"Linear Regression Prediction: {linear_prediction[0]}")