import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("IBEX35_Sept2018.xls")

print(df.head())

x = df['Dia']
y = df['Apertura']

colors = ['red', 'orange', 'green']
lw = 2

x_plot = np.linspace(min(x), max(x), 100)

plt.scatter(x, y, color='navy', s=30, marker='o', label="Puntos de entrenamiento")

print("Ajuste de Regresión polinómica")
for count, degree in enumerate([3, 4, 5]):
    coeffs = np.polyfit(x, y, deg=degree)
    
    p = np.poly1d(coeffs)
    print(f"Polinomio de grado {degree}:")
    print(p)
    print("")
    
    y_pred = np.polyval(p, x)
    ECM = np.mean((y - y_pred)**2)
    print(f"Error cuadrático medio (ECM) para grado {degree}: {ECM}")
    print("")

    y_plot = np.polyval(p, x_plot)
    
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw, label=f"Grado {degree}")

plt.title("Regresión Polinómica para IBEX 35 - Septiembre 2018")
plt.xlabel("Día")
plt.ylabel("Valor de Apertura")
plt.legend(loc='lower left')

plt.show()

coeffs = np.polyfit(x, y, deg=5)
y_pred_dia_6 = np.polyval(np.poly1d(coeffs), 6)
print(f"Predicción para el día 6: {y_pred_dia_6}")
