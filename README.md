# 💳 Detección de Fraudes en Transacciones Financieras con Keras y SMOTE 🧠

## 🚀 Descripción General del Proyecto

Este proyecto se enfoca en la construcción y evaluación de un modelo de **aprendizaje profundo (Deep Learning)** utilizando **Keras** y **TensorFlow** para identificar transacciones financieras fraudulentas. Se aborda el desafío del **alto desbalance de clases** en el dataset mediante la técnica de sobremuestreo **SMOTE** (Synthetic Minority Over-sampling Technique) y se analizan sus efectos en diversas métricas de evaluación.

## 🎯 El Problema

El fraude en transacciones con tarjetas de crédito es un problema crítico que causa pérdidas significativas. Los principales desafíos son:
1.  Identificar con alta precisión las transacciones fraudulentas, que son eventos infrecuentes.
2.  Minimizar los falsos positivos para no incomodar a los clientes con transacciones legítimas bloqueadas.
3.  Manejar eficazmente el **fuerte desbalance de clases** presente en los datos.

## 📊 Dataset Utilizado

Se utilizó el conocido dataset `creditcard.csv`, un estándar para este tipo de problemas:
*   Transacciones de tarjetas de crédito europeas de septiembre de 2013.
*   **Desbalance extremo:** Solo el **0.173%** de las transacciones son fraudulentas.
*   Características `V1` a `V28` anonimizadas mediante PCA.
*   Características originales: `Time` (descartada) y `Amount`.
*   Variable objetivo `Class`: `1` para fraude, `0` para legítima.

## ⚙️ Metodología y Flujo de Trabajo

El proyecto se desarrolló siguiendo estos pasos:

1.  **Carga y Exploración Inicial (EDA):**
    *   Importación de librerías (TensorFlow, Keras, Pandas, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn).
    *   Carga del dataset y visualización de la distribución de clases.

2.  **Preprocesamiento de Datos:**
    *   Eliminación de la columna `Time`.
    *   Separación de características (`X`) y variable objetivo (`y`).
    *   División en conjuntos de **entrenamiento (80%)** y **prueba (20%)**.
    *   **Escalado de características** con `StandardScaler` en los conjuntos de entrenamiento y prueba.

3.  **Modelo Inicial (Sin Manejo de Desbalance Explícito):**
    *   Se construyó, entrenó y evaluó un primer modelo de red neuronal para establecer una línea base.
        *   **Arquitectura:** Entrada -> Densa(16, ReLU, L2=0.1) -> BatchNormalization -> Dropout(0.5) -> Densa(1, Sigmoid).
        *   **Entrenamiento:** Optimizador Adam, pérdida `binary_crossentropy`, `EarlyStopping` (patience=10).
        *   **Evaluación Inicial:** Matriz de confusión y Curva ROC-AUC (umbral de predicción 0.6).

4.  **Manejo del Desbalance de Clases con SMOTE:**
    *   Se aplicó **SMOTE** (`sampling_strategy='auto'`) **únicamente al conjunto de entrenamiento escalado** para generar muestras sintéticas de la clase minoritaria (fraude), equilibrando así las clases para el entrenamiento.

5.  **Reentrenamiento y Evaluación del Modelo con Datos Balanceados:**
    *   Se **reentrenó** un modelo con una arquitectura similar (ajustando la regularización L2 a 0.01 y `EarlyStopping` patience a 15) utilizando los datos de entrenamiento aumentados con SMOTE.
    *   Se evaluó este nuevo modelo en el **conjunto de prueba original (no modificado por SMOTE)** para una evaluación imparcial.
    *   Se generó el **Reporte de Clasificación** (Precisión, Recall, F1-score) y una nueva **Matriz de Confusión** y **Curva ROC-AUC** (umbral de predicción 0.5).

## 🔑 Técnicas y Conceptos Clave Aplicados

*   **Redes Neuronales con Keras/TensorFlow:** Diseño e implementación de arquitecturas de clasificación.
*   **Clasificación Binaria.**
*   **Preprocesamiento de Datos:** Escalado y división.
*   **Regularización:** L2 y Dropout para combatir el sobreajuste.
*   **Batch Normalization:** Para mejorar la convergencia.
*   **Early Stopping:** Para optimizar el entrenamiento.
*   **Manejo del Desbalance de Clases:**
    *   Análisis del impacto del desbalance.
    *   Aplicación de **SMOTE** (Synthetic Minority Over-sampling Technique).
*   **Métricas de Evaluación Robustas:**
    *   Matriz de Confusión.
    *   Curva ROC y Área Bajo la Curva (AUC).
    *   Precisión, Recall y F1-Score (especialmente para la clase minoritaria).

## 📈 Resultados y Evaluación

#### Modelo Inicial (Sin SMOTE, Umbral 0.6):
*   **AUC:** 0.9718. Un buen poder discriminatorio general.
*   La Matriz de Confusión inicial mostró una tendencia a clasificar bien la clase mayoritaria, con un número limitado de fraudes detectados correctamente.

#### Modelo Reentrenado con SMOTE (Umbral 0.5):
*   **AUC:** 0.9651.
*   **Reporte de Clasificación para la Clase Fraude (1):**
    *   **Precisión:** 0.1963
    *   **Recall (Sensibilidad):** 0.8947
    *   **F1-Score:** 0.3220

## ⚖️ Discusión sobre el Impacto de SMOTE

*   **Mejora del Recall:** La aplicación de SMOTE **aumentó significativamente el Recall** para la clase minoritaria (fraude) a **0.89**. Esto significa que el modelo es mucho mejor detectando los fraudes existentes.
*   **Disminución de la Precisión:** Como es común con el sobremuestreo, la **Precisión para la clase fraude disminuyó**. Esto se debe a que al generar más ejemplos de la clase minoritaria, el modelo puede volverse más propenso a clasificar incorrectamente ejemplos de la clase mayoritaria como minoritarios (falsos positivos).
*   **Trade-off Precisión-Recall:** Existe un claro compromiso. En la detección de fraudes, un alto Recall suele ser prioritario, pero esto debe equilibrarse con el coste de los falsos positivos.
*   **Ligera Disminución del AUC:** El AUC bajó ligeramente de 0.9718 a 0.9651. Esto puede suceder porque SMOTE, al generar puntos sintéticos, podría introducir algo de "ruido". Sin embargo, el AUC sigue siendo muy alto.

SMOTE ayudó a que el modelo detectara mejor la clase minoritaria, aunque con un coste en la precisión.

## 💻 Tecnologías Utilizadas

*   ![Python](https://img.shields.io/badge/Python-3.x-blue.svg?style=flat-square&logo=python)
*   ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?style=flat-square&logo=tensorflow)
*   ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=flat-square&logo=Keras&logoColor=white)
*   ![Pandas](https://img.shields.io/badge/Pandas-%23150458.svg?style=flat-square&logo=pandas&logoColor=white)
*   ![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?style=flat-square&logo=numpy&logoColor=white)
*   ![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white)
*   ![Imbalanced-learn](https://img.shields.io/badge/Imbalanced--learn-brightgreen.svg?style=flat-square)
*   ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat-square&logo=Matplotlib&logoColor=black)
*   ![Seaborn](https://img.shields.io/badge/Seaborn-%23023e8a.svg?style=flat-square&logo=seaborn&logoColor=white)
*   Jupyter Notebook / Google Colab

## ✨ Conclusión

Este proyecto demuestra la aplicación de Deep Learning para la detección de fraudes, con un enfoque específico en el manejo del desbalance de clases mediante SMOTE. Se observa que SMOTE puede mejorar significativamente la detección de la clase minoritaria (Recall), aunque introduce un trade-off con la Precisión. El modelo reentrenado con datos balanceados por SMOTE sigue mostrando un alto poder discriminatorio (AUC ~0.97), evidenciando la robustez del enfoque y la importancia de seleccionar métricas adecuadas para problemas desbalanceados.
