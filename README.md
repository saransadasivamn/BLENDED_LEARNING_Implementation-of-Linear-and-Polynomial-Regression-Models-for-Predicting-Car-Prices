# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection: Import essential libraries like pandas, numpy, sklearn, matplotlib, and seaborn. Load the dataset using pandas.read_csv().
2. Data Preprocessing: Address any missing values in the dataset. Select key features for training the models. Split the dataset into training and testing sets with train_test_split().
3. Linear Regression: Initialize the Linear Regression model from sklearn. Train the model on the training data using .fit(). Make predictions on the test data using .predict(). Evaluate model performance with metrics such as Mean Squared Error (MSE) and the R² score.
4. Polynomial Regression: Use PolynomialFeatures from sklearn to create polynomial features. Fit a Linear Regression model to the transformed polynomial features. Make predictions and evaluate performance similar to the linear regression model.
5. Visualization: Plot the regression lines for both Linear and Polynomial models. Visualize residuals to assess model performance

## Program:

Program to implement Linear and Polynomial Regression models for predicting car prices.
Developed by: Arunjuthan.M.A
RegisterNumber:  212225230020

```py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt

df=pd.read_csv('encoded_car_data (1).csv')
print(df.head())

x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

lr= Pipeline([
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])

lr.fit(x_train,y_train)
y_pred_linear = lr.predict(x_test)

poly_model =Pipeline([
    ('poly',PolynomialFeatures(degree=2)),
    ('scaler',StandardScaler()),
    ('model',LinearRegression())
])

poly_model.fit(x_train,y_train)
y_pred_poly=poly_model.predict(x_test)

print("Name: Arunjuthan.M.A")
print("Reg. No: 212225230020")
print()

print("Linear Regression:")
print("MSE=", mean_squared_error(y_test,y_pred_linear))
print("MAE=", mean_absolute_error(y_test,y_pred_linear))
print("R2 score =", r2_score(y_test,y_pred_linear))
print()

print("Polynomial Regression:")
print("MSE=", mean_squared_error(y_test,y_pred_poly))
print("MAE=", mean_absolute_error(y_test,y_pred_poly))
print("R2 score =", r2_score(y_test,y_pred_poly))

plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred_linear,label="Linear",alpha=0.6)
plt.scatter(y_test,y_pred_poly,label='Polynomial (degree=2)',alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--',label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()

```


## Output:
<img width="1382" height="480" alt="image" src="https://github.com/user-attachments/assets/6e58252e-1e24-441e-914e-defe74b5dbcb" />
<img width="348" height="258" alt="image" src="https://github.com/user-attachments/assets/752fbd2d-3b9a-49a4-b408-b4f6e82a9092" />

<img width="1375" height="586" alt="image" src="https://github.com/user-attachments/assets/932b1b51-f8f0-47ec-b627-506966fc0093" />


## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
