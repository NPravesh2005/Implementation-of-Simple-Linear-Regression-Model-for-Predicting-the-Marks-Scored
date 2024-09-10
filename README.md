# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: N.PRAVESH
RegisterNumber:  212223230154
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
df.head()

![image](https://github.com/user-attachments/assets/880e2e5a-5d0b-44e3-8909-ef989b8fa10c)

df.tail()

![image](https://github.com/user-attachments/assets/7aece817-8404-469d-8aff-9345258cf78f)

Array value of x

![image](https://github.com/user-attachments/assets/7aa2ebbb-1147-4b62-a047-9ccbceae7acd)

Array value of y

![image](https://github.com/user-attachments/assets/b9242e42-605d-453f-afe9-a8249cedd927)

Values of y prediction

![image](https://github.com/user-attachments/assets/7d3fd608-18f6-466a-8b54-d6ab8fea73d6)

Array values of y test

![image](https://github.com/user-attachments/assets/97fa9f79-5ec0-4ed2-b858-02fe151f8bee)

Training set graph

![image](https://github.com/user-attachments/assets/282ad575-f4d2-441b-a1d9-b1d6f66552f2)

Test set graph

![image](https://github.com/user-attachments/assets/5f549a4d-3b05-4d53-a30e-4ba26b9f121f)

Value of MSE, MAE, RMSE

![image](https://github.com/user-attachments/assets/35a945bf-da1b-4b1a-8ec8-0b22350c08b4)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
