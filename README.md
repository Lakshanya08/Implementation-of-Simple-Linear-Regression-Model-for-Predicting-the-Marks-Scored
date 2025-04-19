# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.import the needed packages.
2.Assigning hours to x and scores to y.
3.Plot the scatter plot.
4.Use mse,rmse,mae formula to find the values.

## Program:
Name : Lakshanya.N
Reg No : 212224230136
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```
```
df = pd.read_csv("/content/student_scores.csv")
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
```
```
x = df.iloc[:, :-1].values
print(x)
y = df.iloc[:, 1].values
print(y)
```
```
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/3, random_state=0
)
```
```
regressor = LinearRegression()
regressor.fit(x_train, y_train)
```
```
y_pred = regressor.predict(x_test)
print("Predicted values:", y_pred)
print("Actual values:", y_test)
```
```
plt.scatter(x_train, y_train, color='black')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
plt.scatter(x_test, y_test, color='black')
plt.plot(x_train, regressor.predict(x_train), color='red')  # line stays the same
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
mse = mean_absolute_error(y_test, y_pred)
print('MSE =', mse)
mae = mean_absolute_error(y_test, y_pred)
print('MAE =', mae)
rmse = np.sqrt(mse)
print("RMSE =", rmse)
```

## Output:

![image](https://github.com/user-attachments/assets/8cd969a3-525a-4d08-ae69-ea2458873df6)
![image](https://github.com/user-attachments/assets/4075e7ec-5f3b-4bc3-bdbf-1ecc81569d4b)
![image](https://github.com/user-attachments/assets/b3576ebe-72a5-47c4-a571-7dd1ed671b23)
![image](https://github.com/user-attachments/assets/dfd8e994-3d54-46a3-b874-1201137be97b)
![image](https://github.com/user-attachments/assets/a359937e-9fad-4473-9616-e3e91cd6b5af)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
