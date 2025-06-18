# Análisis y Predicción de Ventas en una Tienda de Retail

Este proyecto realiza un análisis exhaustivo de los datos de ventas de una tienda de retail. El objetivo principal es limpiar, transformar y analizar los datos para extraer insights valiosos que puedan ayudar a comprender mejor el comportamiento de las ventas y, potencialmente, a realizar predicciones.

## Estructura del Proyecto

El análisis se divide en varias etapas clave, implementadas en el script `script.py`:

1.  **Carga de Datos**: Se carga el dataset `dirty_cafe_sales.csv`.
2.  **Exploración Inicial de Datos**: Se realiza una inspección básica de los datos para entender su estructura, tipos de datos y estadísticas descriptivas.
3.  **Limpieza de Datos**:
    *   Manejo de valores nulos mediante interpolación para columnas numéricas y de fecha, y `ffill`/`bfill` para categóricas.
    *   Identificación y conteo de datos duplicados.
4.  **Transformación de Datos**:
    *   Creación de nuevas columnas:
        *   `Ingreso`: Calculado como `Quantity * Price Per Unit`.
        *   `Categoria Ingreso`: Clasificación del ingreso en 'Bajo', 'Medio', 'Alto'.
        *   `Categoria Precio`: Clasificación del precio unitario en 'Bajo', 'Medio', 'Alto'.
        *   `Dia Semana`: Extraído de `Transaction Date`.
        *   `Mes`: Extraído de `Transaction Date`.
    *   Normalización (Min-Max scaling) y Estandarización (Z-score scaling) de columnas numéricas (`Quantity`, `Price Per Unit`, `Ingreso`).
5.  **Análisis de Datos**:
    *   **Agrupación y Agregación**:
        *   Ventas totales por producto (`Item`), región (`Location`), día de la semana y mes.
        *   Número de transacciones por producto.
        *   Cálculo de estadísticas descriptivas (suma, media, conteo, mínimo, máximo, desviación estándar, varianza) para el `Ingreso` agrupado por `Item`.
    *   **Análisis Personalizado con `apply`**:
        *   Cálculo del rango de `Ingreso` por `Item`.
        *   Creación de una columna `Tipo Transaccion Compleja` basada en condiciones personalizadas sobre `Ingreso`, `Quantity` y `Payment Method`.
        *   Cálculo de la desviación de cada venta (`Ingreso`) respecto a la media de su grupo (`Item`) utilizando `transform`.

## Requisitos

*   Python 3.x
*   pandas
*   numpy
*   scikit-learn

Puedes instalar las librerías necesarias usando pip:
```bash
pip install pandas numpy scikit-learn
