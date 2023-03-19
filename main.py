# Import the necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

# Load the dataset
data = pd.read_csv('data.csv')

# Check for missing values and fill them with the mean of the column
if data.isna().any().any():
    data = data.fillna(data.mean())

# Split the dataset into input and output variables
X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create an instance of the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Evaluate the model on the testing data
score = model.score(X_test, y_test)
print("R-squared score: ", score)

# Use cross-validation to evaluate the model's performance
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores: ", cv_scores)
print("Average cross-validation score: ", np.mean(cv_scores))
