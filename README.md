## #Proyecto de Predicción de la Huella de Carbono

###### Este proyecto utiliza un modelo de Machine Learning (Random Forest) para predecir la huella de carbono generada por diferentes actividades. Además, identifica los factores más significativos que contribuyen a estas emisiones. El proyecto se ejecuta en un entorno de Google Colab y proporciona una interfaz sencilla para la carga de datos y la visualización de resultados.

## Instrucciones para Ejecutar el Proyecto
**1. Abrir el Proyecto en Google Colab**
Haz clic en el siguiente enlace para abrir el proyecto en Google Colab:

https://colab.research.google.com/drive/14itWi7iO1dZRPBDZAncPs473nI7i8I0n#scrollTo=hfHUbvRLzp1-

**2. Instalar Dependencias**
Antes de ejecutar el código principal, asegúrate de instalar la biblioteca ipywidgets, que es necesaria para la interfaz de carga de archivos. Ejecuta la siguiente celda en Colab:

*!pip install ipywidgets*

**3. Ejecutar el Código Principal**
Después de instalar las dependencias, ejecuta el código principal en la celda correspondiente en Google Colab. Este código te permitirá cargar un archivo CSV con los datos y realizar el análisis de predicción de la huella de carbono.

**4. Cargar el Dataset**
Puedes utilizar el siguiente dataset de pruebas para cargar los datos en la aplicación:

https://drive.google.com/file/d/1cQYwKsLXiLD9poQRWU9KCutPpv7UZXxC/view?usp=sharing

Sube este archivo CSV cuando se te solicite en el entorno de Colab.

**5. Visualizar Resultados**
Una vez que hayas cargado el archivo, el código realizará el análisis de predicción y mostrará los resultados, incluyendo el Coeficiente de Determinación (R²), el Error Cuadrático Medio (MSE), y la importancia de las características en la predicción de las emisiones de carbono.

**🧪 Pruebas Unitarias**
El proyecto incluye pruebas unitarias para verificar que las funciones principales del modelo funcionan correctamente. Las pruebas están integradas en el código y se ejecutan automáticamente al final de la ejecución.

**Pruebas Incluidas:**
1. Prueba de Entrenamiento del Modelo:
Verifica que el modelo se entrena correctamente y que el valor de R² es mayor que 0.

2. Prueba de Importancia de Características:
Verifica que la función de importancia de características retorne un DataFrame con las columnas Feature e Importance.

**📄 Documentación Adicional**
Para más detalles sobre el proyecto, puedes consultar la siguiente documentación:

https://docs.google.com/document/d/1Vj379WNFH9f9Pw1kspXIY635f7fHroMA2FrLPKIlQjk/edit?usp=sharing
