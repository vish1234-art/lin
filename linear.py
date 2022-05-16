import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("hours.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
print("Accuracy : ", str(regressor.score(X, y) * 100))
y_pred=regressor.predict([[10]])
print(y_pred)
hours=int(input('Enter the no of hours:'))
eq=regressor.coef_*hours+regressor.intercept_
print('y = %f*%f+%f' %(regressor.coef_,hours,regressor.intercept_))
print('Equation for best fit line is: y=%f*x+%f' %(regressor.coef_,regressor.intercept_))
print("Risk Score: ", eq[0])
plt.plot(X, y, 'o')
plt.plot(X, regressor.predict(X))
plt.show()
