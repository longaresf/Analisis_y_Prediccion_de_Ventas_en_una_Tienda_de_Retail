# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Cargar los Datos del dataset
path = 'dataset/dirty_cafe_sales.csv'

df_csv = pd.read_csv(path)

# Mostrar las primeras 10 filas
print("\nPrimeras 10 filas del DataFrame:")
print(df_csv.head(10))

# 3. Exploración Inicial de los Datos
# Mostrar las últimas 5 filas
print("\nÚltimas 5 filas del DataFrame:")
print(df_csv.tail())

# Información general del DataFrame
print("\nInformación general del DataFrame:")
df_csv.info()

# Estadísticas descriptivas del DataFrame
print("\nEstadísticas descriptivas del DataFrame:")
print(df_csv.describe())

# 4. Limpieza de Datos
# Identificar valores nulos en todo el DataFrame
print("\nIdentificar valores nulos en todo el DataFrame:")
print(df_csv.isnull().sum())

# Rellenar valores nulos con interpolación
# Se utiliza una copia del dataframe
df_interpolate=df_csv.copy()

# Identificar valores nulos en columna 'Quantity'
print("\nIdentificar valores nulos en columna 'Quantity':")
print(df_interpolate['Quantity'].isnull().sum())

# Se rellena los valores vacios con Nan y se interpola
df_interpolate['Quantity'] = pd.to_numeric(df_interpolate['Quantity'], errors='coerce').interpolate()


# Identificar valores nulos en columna 'Price Per Unit'
print("\nIdentificar valores nulos en columna 'Price Per Unit':")
print(df_interpolate['Price Per Unit'].isnull().sum())

# Se rellena los valores vacios con Nan y se interpola
df_interpolate['Price Per Unit'] = pd.to_numeric(df_interpolate['Price Per Unit'], errors='coerce').interpolate()


# Identificar valores nulos en columna 'Total Spent'
print("\nIdentificar valores nulos en columna 'Total Spent':")
print(df_interpolate['Total Spent'].isnull().sum())

# Se rellena los valores vacios con Nan y se interpola
df_interpolate['Total Spent'] = pd.to_numeric(df_interpolate['Total Spent'], errors='coerce').interpolate()


# Identificar valores nulos en columna 'Transaction Date'
print("\nIdentificar valores nulos en columna 'Transaction Date':")
print(df_interpolate['Transaction Date'].isnull().sum())

# Se convierte 'Transaction Date' a objetos datetime, manejando errores y formatos mixtos.
df_interpolate['Transaction Date'] = pd.to_datetime(df_interpolate['Transaction Date'], errors='coerce')
# Se interpolan los valores NaT en la columna 'Transaction Date'.
df_interpolate['Transaction Date'] = df_interpolate['Transaction Date'].interpolate()


# Identificar valores nulos en columna 'Item'
print("\nIdentificar valores nulos en columna 'Item':")
print(df_interpolate['Item'].isnull().sum())

# Se rellena los valores vacios con Nan y se interpola
df_interpolate['Item'] = df_interpolate['Item'].ffill()


# Identificar valores nulos en columna 'Payment Method'
print("\nIdentificar valores nulos en columna 'Payment Method':")
print(df_interpolate['Payment Method'].isnull().sum())

# Se rellena los valores vacios con Nan y se interpola
df_interpolate['Payment Method'] = df_interpolate['Payment Method'].bfill()


# Identificar valores nulos en columna 'Location'
print("\nIdentificar valores nulos en columna 'Location':")
print(df_interpolate['Location'].isnull().sum())

# Se rellena los valores vacios con Nan y se interpola
df_interpolate['Location'] = df_interpolate['Location'].ffill()

# Identificar valores nulos en todo el DataFrame
print("\nIdentificar valores nulos en todo el DataFrame:")
print(df_interpolate.isnull().sum())


# Datos duplicados
print("\nIdentificar valores duplicados en todo el DataFrame:")
print(df_interpolate.duplicated().sum())


# 5. Transformación de Datos
# Tabla de ingresos a partir de ventas y precios
df_interpolate['Ingreso'] = df_interpolate['Quantity'] * df_interpolate['Price Per Unit']
df = df_interpolate[['Transaction ID', 'Quantity', 'Price Per Unit', 'Ingreso']]
print("\nDataFrame con la nueva columna 'Ingreso' (primeras 5 filas):")
print(df.head())

# calcular ingresos a partir de ventas y precios
print("\nIngresos totales:")
print(df_interpolate['Ingreso'].sum().round(2))

# Normaliza o estandariza columnas
# Transformación de Datos Adicional: Normalización y Estandarización
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Seleccionar columnas numéricas para escalar
cols_to_scale = ['Quantity', 'Price Per Unit', 'Ingreso']

# Crear copias del DataFrame para no modificar el original directamente en este paso
df_standardized = df_interpolate.copy()
df_normalized = df_interpolate.copy()

# Estandarización (Z-score scaling)
scaler_standard = StandardScaler()
df_standardized[cols_to_scale] = scaler_standard.fit_transform(df_interpolate[cols_to_scale])

print("\nDataFrame con columnas estandarizadas (primeras 5 filas):")
print(df_standardized[['Transaction ID'] + cols_to_scale].head())

# Normalización (Min-Max scaling)
scaler_minmax = MinMaxScaler()
df_normalized[cols_to_scale] = scaler_minmax.fit_transform(df_interpolate[cols_to_scale])

print("\nDataFrame con columnas normalizadas (primeras 5 filas):")
print(df_normalized[['Transaction ID'] + cols_to_scale].head())


# Clasifica los datos en categorías relevantes.
print("\nClasificación de datos:")

# Categorizar 'Ingreso'
# Definir los límites para las categorías de ingreso
labels_ingreso = ['Ingreso Bajo', 'Ingreso Medio', 'Ingreso Alto']
max_ingreso = df_interpolate['Ingreso'].max()

if pd.isna(max_ingreso):
    bins_ingreso = [0, 1, 2, 3]
elif max_ingreso > 15:
    bins_ingreso = [0, 5, 15, max_ingreso]
elif max_ingreso > 5: 
    bins_ingreso = [0, 5, (5 + max_ingreso) / 2.0, max_ingreso]
elif max_ingreso > 0:
    bins_ingreso = list(np.linspace(0, max_ingreso, 4))
else:
    bins_ingreso = [0, 1, 2, 3]

df_interpolate['Categoria Ingreso'] = pd.cut(df_interpolate['Ingreso'], bins=bins_ingreso, labels=labels_ingreso, right=True, include_lowest=True)

print("\nPrimeras 5 filas con 'Categoria Ingreso':")
print(df_interpolate[['Ingreso', 'Categoria Ingreso']].head())

# Categorizar 'Price Per Unit'
# Definir los límites para las categorías de precio
labels_precio = ['Precio Bajo', 'Precio Medio', 'Precio Alto']
max_price = df_interpolate['Price Per Unit'].max()

if pd.isna(max_price):
    bins_precio = [0, 1, 2, 3]
elif max_price > 6:
    bins_precio = [0, 3, 6, max_price]
elif max_price > 3:
    bins_precio = [0, 3, 3 + (max_price - 3) / 2.0, max_price]
elif max_price > 0:
    bins_precio = list(np.linspace(0, max_price, 4))
else:
    bins_precio = [0, 1, 2, 3]

df_interpolate['Categoria Precio'] = pd.cut(df_interpolate['Price Per Unit'], bins=bins_precio, labels=labels_precio, right=True, include_lowest=True)

print("\nPrimeras 5 filas con 'Categoria Precio':")
print(df_interpolate[['Price Per Unit', 'Categoria Precio']].head())

# Extraer día de la semana de 'Transaction Date'
df_interpolate['Dia Semana'] = df_interpolate['Transaction Date'].dt.day_name()

print("\nPrimeras 5 filas con 'Dia Semana':")
print(df_interpolate[['Transaction Date', 'Dia Semana']].head())

# 6. Análisis de Datos
# Agrupaciones para obtener insights

print("\nAnálisis de Datos Agrupados:")

# Ventas totales por producto (Item)
ventas_por_producto = df_interpolate.groupby('Item')['Ingreso'].sum().sort_values(ascending=False)
print("\nVentas totales por producto:")
print(ventas_por_producto)

# Ventas totales por región (Location)
ventas_por_region = df_interpolate.groupby('Location')['Ingreso'].sum().sort_values(ascending=False)
print("\nVentas totales por región:")
print(ventas_por_region)

# Número de transacciones por producto
transacciones_por_producto = df_interpolate.groupby('Item')['Transaction ID'].count().sort_values(ascending=False)
print("\nNúmero de transacciones por producto:")
print(transacciones_por_producto)


# Ventas totales por día de la semana
ventas_por_dia_semana = df_interpolate.groupby('Dia Semana')['Ingreso'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
print("\nVentas totales por día de la semana:")
print(ventas_por_dia_semana)

# Aplica funciones de agregación como sum, mean, count, min, max, std, y var.
# Aplicar múltiples funciones de agregación
print("\nEstadísticas de Ingreso por Producto:")
ingreso_stats_por_producto = df_interpolate.groupby('Item')['Ingreso'].agg(['sum', 'mean', 'count', 'min', 'max', 'std', 'var'])
print(ingreso_stats_por_producto)

# Utiliza el método apply para realizar operaciones más complejas y personalizadas.
print("\nUso de apply para operaciones personalizadas:")

# Ejemplo 1: apply con groupby para calcular el rango de 'Ingreso' por 'Item'
def calcular_rango_ingreso(grupo):
    # Esta función recibe cada grupo (un DataFrame) del groupby
    return grupo['Ingreso'].max() - grupo['Ingreso'].min()

rango_ingreso_por_producto = df_interpolate.groupby('Item').apply(calcular_rango_ingreso)
print("\nRango de Ingreso (max - min) por Producto (usando groupby.apply):")
print(rango_ingreso_por_producto.sort_values(ascending=False))


# Ejemplo 2: apply en filas (axis=1) para crear una nueva columna basada en una condición compleja
def categorizar_transaccion_compleja(fila):
    if fila['Ingreso'] > 15 and fila['Payment Method'] == 'Credit Card':
        return 'Compra Grande con Tarjeta'
    elif fila['Quantity'] == 1 and fila['Ingreso'] < 5:
        return 'Compra Pequeña Unitaria'
    else:
        return 'Otra Transacción'

df_interpolate['Tipo Transaccion Compleja'] = df_interpolate.apply(categorizar_transaccion_compleja, axis=1)
print("\nPrimeras 5 filas con 'Tipo Transaccion Compleja' (usando df.apply axis=1):")
print(df_interpolate[['Ingreso', 'Quantity', 'Payment Method', 'Tipo Transaccion Compleja']].head())


# Parte III: Transformación y Análisis Avanzado de Datos con Pandas
# 1. Transformación de Datos

# Extraer mes de 'Transaction Date' para análisis temporal
df_interpolate['Mes'] = df_interpolate['Transaction Date'].dt.month_name()
print("\nCálculos con transformación de Datos:")
print("\nPrimeras 5 filas con 'Mes':")
print(df_interpolate[['Transaction Date', 'Mes']].head())


# 2. Agrupación y Agregación
# Agrupaciones para obtener insights

print("\nVentas totales por día de la semana:")
print(ventas_por_dia_semana)

# Ventas totales por producto y mes
ventas_por_producto_mes = df_interpolate.groupby(['Item', 'Mes'])['Ingreso'].sum()
print("\nVentas totales por producto y mes:")
print(ventas_por_producto_mes.head()) # Mostrar solo las primeras para brevedad

# 3. Análisis Personalizado con apply
# Aplica funciones de agregación como sum, mean, count, min, max, std, y var.
# Aplicar múltiples funciones de agregación
print("\nEstadísticas de Ingreso por Producto:")
print("\nPrimeras 5 filas con 'Tipo Transaccion Compleja' (usando df.apply axis=1):")
print(df_interpolate[['Ingreso', 'Quantity', 'Payment Method', 'Tipo Transaccion Compleja']].head())


# Calcula la desviación de cada venta ('Ingreso') respecto a la media de su grupo ('Item'). para identificar ventas atípicas dentro de cada categoría de producto.
print("\nCálculo de la desviación del Ingreso respecto a la media por Item:")
# Se calcula el ingreso medio por 'Item' y se alinea con el DataFrame original
df_interpolate['Ingreso Medio por Item'] = df_interpolate.groupby('Item')['Ingreso'].transform('mean')
# Se calcula la desviación de cada venta respecto a la media de su 'Item'
df_interpolate['Desviacion Ingreso'] = df_interpolate['Ingreso'] - df_interpolate['Ingreso Medio por Item']

print("\nPrimeras 5 filas con 'Desviacion Ingreso' y 'Ingreso Medio por Item':")
print(df_interpolate[['Item', 'Ingreso', 'Ingreso Medio por Item', 'Desviacion Ingreso']].head())


# PARTE IV: Análisis Exploratorio de Datos (EDA) y Visualización

print("\n--- Iniciando Parte IV: Análisis Exploratorio de Datos (EDA) y Visualización ---")

# 1. Análisis Descriptivo
print("\nInformación general del DataFrame:")
df.info()

print("\nEstadísticas descriptivas de las variables numéricas:")
print(df.describe())


# 2. Visualización de Datos

# Configuración de estilo para los gráficos
sns.set_style("whitegrid")

# Combinar histogramas y boxplots para visualizar la distribución de las ventas y detectar valores atípicos.
# Usando la columna 'Ingreso' que representa las ventas por transacción.
print("\nGenerando visualización de la distribución de Ingresos por transacción...")
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, 
                         gridspec_kw={"height_ratios": (.8, .2)})
fig.suptitle('Distribución de Ingresos y Detección de Atípicos', fontsize=16)

# Histograma
sns.histplot(df_interpolate['Ingreso'], ax=axes[0], kde=True, bins=40)
axes[0].set_ylabel('Frecuencia')
axes[0].set_xlabel('')

# Boxplot
sns.boxplot(x=df_interpolate['Ingreso'], ax=axes[1])
axes[1].set_xlabel('Ingreso por Transacción')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# 3. Crear gráficos de líneas para mostrar las tendencias de ventas a lo largo del tiempo.
print("\nGenerando visualización de tendencias de ventas a lo largo del tiempo...")
# Agrupamos los ingresos por semana para visualizar la tendencia.
df_temporal = df_interpolate.set_index('Transaction Date')
ventas_semanales = df_temporal['Ingreso'].resample('W').sum()

plt.figure(figsize=(15, 7))
ventas_semanales.plot(title='Tendencia de Ingresos Semanales', marker='o', linestyle='-')
plt.xlabel('Semana')
plt.ylabel('Ingresos Totales')
plt.grid(True)
plt.tight_layout()
plt.show()


# 4. Utilizar gráficos de dispersión para analizar la relación entre diferentes variables.
# Analizaremos la relación entre 'Price Per Unit' y 'Quantity'.
print("\nGenerando visualización de la relación entre Precio Unitario y Cantidad...")
plt.figure(figsize=(12, 7))
sns.scatterplot(data=df_interpolate, x='Price Per Unit', y='Quantity', alpha=0.6, hue='Categoria Precio')
plt.title('Relación entre Precio Unitario y Cantidad Vendida')
plt.xlabel('Precio Unitario')
plt.ylabel('Cantidad')
plt.grid(True)
plt.tight_layout()
plt.show()
