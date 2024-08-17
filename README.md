
# Análisis de Huella de Carbono con Random Forest

Este proyecto permite analizar las emisiones de carbono utilizando un modelo de Random Forest. La aplicación está diseñada para ser ejecutada con Streamlit y también incluye pruebas unitarias para garantizar la calidad del código.

## Requisitos

- Python 3.8+
- Pip (gestor de paquetes de Python)
- Las siguientes librerías de Python:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```txt
numpy
pandas
scikit-learn
streamlit
matplotlib
unittest
```

## Estructura del Proyecto

- `carbon_emission_app.py`: Contiene el código principal del proyecto, incluyendo la lógica para el análisis y las pruebas unitarias.
- `README.md`: Este archivo, que proporciona instrucciones sobre cómo ejecutar el proyecto y las pruebas.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.

## Ejecución de la Aplicación

Para ejecutar la aplicación, sigue estos pasos:

1. Asegúrate de estar en el directorio donde se encuentra el archivo `carbon_emission_app.py`.
2. Ejecuta el siguiente comando en la terminal:

```bash
streamlit run carbon_emission_app.py
```

3. La aplicación se abrirá automáticamente en tu navegador web en la dirección `http://localhost:8501`.
4. Sube un archivo CSV que contenga los datos a analizar y obtendrás la predicción de emisiones de carbono junto con la importancia de las características. en el repositorio se pone uno de prueba llamado CarbonEmission.csv

## Ejecución de Pruebas Unitarias

Las pruebas unitarias están incluidas en el mismo archivo `carbon_emission_app.py`. Para ejecutarlas:

1. Ejecuta el archivo Python directamente desde la terminal:

```bash
python carbon_emission_app.py
```

2. Las pruebas se ejecutarán automáticamente y los resultados se mostrarán en la terminal. Si todas las pruebas pasan, verás un mensaje de "OK". Si alguna falla, se mostrará un mensaje con detalles sobre la prueba que falló.

## Funciones Principales

- **`preprocess_data(data)`**: Preprocesa los datos, codifica variables categóricas y separa características (`X`) del objetivo (`y`).
- **`train_model(X, y)`**: Entrena un modelo de RandomForest y devuelve el modelo, MSE y R².
- **`feature_importance(regressor, X)`**: Calcula y devuelve la importancia de las características.
- **`analyze_data(uploaded_file)`**: Orquesta todo el proceso de análisis y visualización de resultados.

## Pruebas Unitarias

- **`test_preprocess_data`**: Verifica que la función `preprocess_data` funcione correctamente.
- **`test_train_model`**: Prueba que el modelo se entrene correctamente y que el R² sea mayor que 0.
- **`test_feature_importance`**: Verifica que la función de importancia de características retorne un DataFrame correcto.

