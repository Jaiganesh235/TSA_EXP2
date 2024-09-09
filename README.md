### DEVELOPED BY: S JAIGANESH
### REGISTER NO: 212222240037
### DATE:

# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

## AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

## ALGORITHM:
1. Import necessary libraries (NumPy, Matplotlib)
2. Load the dataset
3. Calculate the linear trend values using least square method
4. Calculate the polynomial trend values using least square method
5. End the program

## PROGRAM:
```python
A - LINEAR TREND ESTIMATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/mnt/data/MLTempDataset.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Group by date and calculate the average 'DAYTON_MW' per day
daily_average = data.groupby(data['Datetime'].dt.date)['DAYTON_MW'].mean().reset_index()

# Linear trend estimation
x = np.arange(len(daily_average))
y = daily_average['DAYTON_MW']
linear_coeffs = np.polyfit(x, y, 1)
linear_trend = np.polyval(linear_coeffs, x)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(daily_average['Datetime'], daily_average['DAYTON_MW'], label='Original Data', marker='o')
plt.plot(daily_average['Datetime'], linear_trend, label='Linear Trend', color='red')
plt.title('Linear Trend Estimation for DAYTON_MW')
plt.xlabel('Date')
plt.ylabel('Average DAYTON_MW')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



B- POLYNOMIAL TREND ESTIMATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('/mnt/data/MLTempDataset.csv')
data['Datetime'] = pd.to_datetime(data['Datetime'])

# Group by date and calculate the average 'DAYTON_MW' per day
daily_average = data.groupby(data['Datetime'].dt.date)['DAYTON_MW'].mean().reset_index()

# Polynomial trend estimation (degree 2)
x = np.arange(len(daily_average))
y = daily_average['DAYTON_MW']
poly_coeffs = np.polyfit(x, y, 2)
poly_trend = np.polyval(poly_coeffs, x)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(daily_average['Datetime'], daily_average['DAYTON_MW'], label='Original Data', marker='o')
plt.plot(daily_average['Datetime'], poly_trend, label='Polynomial Trend (Degree 2)', color='green')
plt.title('Polynomial Trend Estimation for DAYTON_MW')
plt.xlabel('Date')
plt.ylabel('Average DAYTON_MW')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



```
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## OUTPUT:
A - LINEAR TREND ESTIMATION
![image](https://github.com/user-attachments/assets/8ddb27df-8542-436e-a5de-3f26b033b5c5)



B- POLYNOMIAL TREND ESTIMATION
![image](https://github.com/user-attachments/assets/8c12d074-4a04-4f3e-8533-a879970e739b)



## RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
