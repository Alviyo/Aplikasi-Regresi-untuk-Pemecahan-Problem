import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial

# Membaca data dari file Excel
file_path = "C:\\Users\\ASUS\\Downloads\\archive\\Student_Performance.csv"
df = pd.read_csv(file_path)

# Memeriksa lima baris pertama dataset
print(df.head())

# Problem 1: Durasi waktu belajar (TB) terhadap nilai ujian (NT)
X = df['Hours_Studied'].values.reshape(-1, 1)
y = df['Performance_Index'].values

# Metode 1: Model Linear
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
rms_linear = np.sqrt(mean_squared_error(y, y_pred_linear))

# Metode 2: Model Pangkat Sederhana
def power_law(x, a, b):
    return a * np.power(x, b)

params, _ = curve_fit(power_law, df['Hours_Studied'], df['Performance_Index'])
y_pred_power = power_law(df['Hours_Studied'], *params)
rms_power = np.sqrt(mean_squared_error(y, y_pred_power))

# Metode Opsional: Model Polinomial
# Degree 2 polynomial fit
p = Polynomial.fit(df['Hours_Studied'], df['Performance_Index'], 2)
y_pred_poly = p(df['Hours_Studied'])
rms_poly = np.sqrt(mean_squared_error(y, y_pred_poly))

# Plot hasil regresi
plt.figure(figsize=(12, 6))

# Plot data asli
plt.scatter(df['Hours_Studied'], df['Performance_Index'], color='blue', label='Data Asli')

# Plot Model Linear
plt.plot(df['Hours_Studied'], y_pred_linear, color='red', label=f'Model Linear (RMS={rms_linear:.2f})')

# Plot Model Pangkat Sederhana
plt.plot(df['Hours_Studied'], y_pred_power, color='green', label=f'Model Pangkat Sederhana (RMS={rms_power:.2f})')

# Plot Model Polinomial
plt.plot(df['Hours_Studied'], y_pred_poly, color='purple', label=f'Model Polinomial (RMS={rms_poly:.2f})')

plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.legend()
plt.title('Regresi Hours Studied terhadap Performance Index')
plt.show()

print(f'RMS Model Linear: {rms_linear:.2f}')
print(f'RMS Model Pangkat Sederhana: {rms_power:.2f}')
print(f'RMS Model Polinomial: {rms_poly:.2f}')
