# Predicción de compra en una red social · Regresión logística

Proyecto de la asignatura **Inteligencia Artificial** (Tema 3 · Clasificación) — **Grupo 6**: Mario Ibañez, Lorenzo Torres y Pablo Piqueras.

El departamento de marketing dispone de datos de usuarios de una red social y de si compraron o no un producto anunciado. El objetivo es **analizar qué característi
cas influyen en la compra** y construir un **modelo de regresión logística** que prediga si un usuario comprará.

## Dataset

`Social_Network_Ads.csv` · 400 usuarios, 5 columnas:

| Variable | Descripción | Uso |
|---|---|---|
| `User ID` | Identificador del usuario | Se excluye del modelo |
| `Gender` | Male / Female | Predictor (codificado 0/1) |
| `Age` | Edad en años | Predictor |
| `EstimatedSalary` | Salario anual estimado | Predictor |
| `Purchased` | 0 = no compra, 1 = compra | Variable objetivo |

Sin nulos ni duplicados. Clases desbalanceadas: 64 % no compra y 36 % compra.

## Qué hace el notebook

`Social_Network_Ads_Regresion_Logistica.ipynb` está estructurado como un informe:

1. **Entorno**: librerías y semilla fija (42).
2. **Auditoría**: dimensiones, tipos, nulos, duplicados, valores incorrectos y desbalance.
3. **Partición**: codificación de `Gender` y división Train/Test 75/25 estratificada.
4. **EDA sobre Train**: estadísticas descriptivas, histogramas, boxplots, matriz de correlación y pairplot.
5. **Preparación y modelos**: estandarización dentro de un `Pipeline` (sin fuga de datos) y 10 regresiones logísticas candidatas.
6. **Validación cruzada**: 5 particiones × 3 repeticiones, solo sobre Train, ajustando la regularización `C`.
7. **Evaluación final**: matrices de confusión, Exactitud, Sensibilidad, Especificidad y Precisión en Train y Test, con la justificación de cada modelo según la teo
ría.
8. **Visualización**: probabilidades predichas en Train, fronteras de decisión y coeficientes.
9. **Conclusiones y límites.**

### Modelos comparados

| | Modelo | | Modelo |
|---|---|---|---|
| M1 | Género | M6 | Interacción edad × salario |
| M2 | Edad | M7 | Edad + salario, cuadrático |
| M3 | Salario | M8 | Cuadrático + género |
| M4 | Edad + salario | M9 | Edad + salario, balanceado |
| M5 | Edad + salario + género | M10 | Cuadrático, balanceado |

### Criterio de selección

Para marketing, perder un comprador (falso negativo) cuesta más que mostrar el anuncio a quien no compra (falso positivo). Por eso se **prioriza la Sensibilidad** y
 se **desempata por Exactitud**, vigilando que la Especificidad y la Precisión no caigan.

## Resultados

Modelo elegido: **M10 · edad y salario con términos cuadráticos, `class_weight='balanced'` y C = 10**.

| | Exactitud | Sensibilidad | Especificidad | Precisión |
|---|---|---|---|---|
| Validación cruzada | 90,44 % | 90,74 % | 90,33 % | 84,34 % |
| Test | 89,00 % | 86,11 % | 90,62 % | 83,78 % |

En Test detecta **31 de los 36 compradores**, con 6 falsas alarmas. La diferencia entre Train y Test es pequeña, así que no hay indicios de overfitting fuerte.

**Conclusiones principales:**
- La **edad** es la variable más informativa; combinada con el **salario** mejora claramente.
- La relación **no es lineal**: los modelos con interacción o términos cuadráticos superan al lineal.
- El **género** no aporta capacidad predictiva.

## Cómo ejecutarlo

Requisitos: Python 3 con `pandas`, `numpy`, `matplotlib`, `seaborn` y `scikit-learn` (versión reciente; probado con scikit-learn 1.9.1).

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

1. Coloca `Social_Network_Ads.csv` en la misma carpeta que el notebook (o ajusta la ruta en la celda de carga).
2. Abre `Social_Network_Ads_Regresion_Logistica.ipynb` en Jupyter, PyCharm o Google Colab.
3. Selecciona **Reiniciar kernel y ejecutar todas las celdas**.

El notebook ya incluye las salidas de una ejecución completa.
