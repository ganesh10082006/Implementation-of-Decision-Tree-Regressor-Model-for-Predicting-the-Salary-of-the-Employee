# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: GANESH G.
RegisterNumber:  212223230059
*/

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Read the dataset
data = pd.read_csv("/content/Salary_EX7.csv")

# Display the first few rows of the dataset
data.head()

# Display information about the dataset
data.info()

# Check for missing values
data.isnull().sum()

# Encode the 'Position' column
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

# Display the first few rows of the modified dataset
data.head()

# Define features and target variable
x = data[["Position", "Level"]]
y = data["Salary"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Initialize the DecisionTreeRegressor
dt = DecisionTreeRegressor()

# Train the model
dt.fit(x_train, y_train)

# Make predictions
y_pred = dt.predict(x_test)

# Calculate Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)

# Print Mean Squared Error
print("Mean Squared Error:", mse)

# Calculate R^2 score
r2 = metrics.r2_score(y_test, y_pred)

# Print R^2 score
print("R^2 Score:", r2)

# Predict on a new data point
prediction = dt.predict([[5, 6]])
print("Predicted salary:", prediction)

```

## Output:

### 1. DATA:
![image](https://github.com/user-attachments/assets/2efad319-db29-40d7-89f8-3b9020ee17e4)

### 2. MEAN SQUARED ERROR:
![image](https://github.com/user-attachments/assets/dd7afbf2-0ba7-4bca-b7ed-7930b4c1b56c)

### 3. PREDICTION:
![image](https://github.com/user-attachments/assets/8bf8e016-924d-474b-9a21-b98fe5010cb8)

### 4.CART ALGORITHM:
![image](https://github.com/user-attachments/assets/160908bc-ce25-4352-b00e-518847f6629c)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
