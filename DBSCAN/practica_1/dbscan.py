##########################################################################
################### Clustering con DBSCAN ################################
##########################################################################

### Librerias básicas
import pandas as pd                 # Trabajar con bases de datos
import numpy as np                  # Trabajar con multidimensionalidad


###### Loading and Plot Data
import matplotlib.pyplot as plt     # Trabajar con exploración gráfica
from pandas import read_excel
from pathlib import Path

data_base = pd.read_excel('Odd_shape_data.xls')
print(f'Los primeros casos son:\n {data_base.head()}')

### 1. Representación Gráfica.

X = data_base.iloc[:, :2]

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], color="navy")
plt.title("Data")
plt.xlabel("V1")
plt.ylabel("V2")
plt.show()

#### 2. Estimación del modelo 
from sklearn.cluster import DBSCAN      # Modelo de clusterización con densidad.

def modelo(e, n):

    model = DBSCAN(eps=e, min_samples=n).fit(X)
    labels = model.labels_
    noise_mask = labels == -1

    n_clusters = len(set(labels) - {-1})
    n_noise = np.sum(noise_mask)

    print('estimated no. of clusters: %d' % n_clusters)
    print('estimated no. of noise points: %d' % n_noise)

    # Scatter plot con outliers
    colors = list(map(lambda x: 'red' if x == -1 else 'brown' if x == 0 else 'cyan', labels))

    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors, marker='o', picker=True)
    plt.scatter(X.iloc[noise_mask, 0], X.iloc[noise_mask, 1],
                c='red', marker='x', label='Noisy Points')
    plt.title('DBSCAN Clustering')
    plt.xlabel('Axis V1')
    plt.ylabel('Axis V2')
    plt.legend()
    plt.show()

    # Remove the noise
    X_sin_ruido = X.iloc[~noise_mask].to_numpy()
    labels_sin_ruido = labels[~noise_mask]

    # Scatter plot sin outliers
    colors = list(map(lambda x: 'red' if x == -1 else 'brown' if x == 0 else 'cyan', labels_sin_ruido))

    plt.scatter(X_sin_ruido[:, 0], X_sin_ruido[:, 1],
                c=colors, marker='o', picker=True)
    plt.title('DBSCAN Clustering sin ruido')
    plt.xlabel('Axis V1')
    plt.ylabel('Axis V2')
    plt.show()


modelo(e= float(input('¿Radio de distacia (epsilon)?')),
       n = int(input('¿Número de muestra?')))



