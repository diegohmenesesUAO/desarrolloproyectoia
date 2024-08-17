import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import unittest

# Función para preprocesar los datos
def preprocess_data(data):
    categorical_columns = data.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])
    X = data.drop(columns=['CarbonEmission'])
    y = data['CarbonEmission']
    return X, y

# Función para entrenar el modelo
def train_model(X, y):
    regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
    regressor.fit(X, y)
    predictions = regressor.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    return regressor, mse, r2

# Función para calcular la importancia de las características
def feature_importance(regressor, X):
    feature_importances = regressor.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    return feature_importance_df

# función que encapsula todo el análisis
def analyze_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    X, y = preprocess_data(data)
    regressor, mse, r2 = train_model(X, y)
    feature_importance_df = feature_importance(regressor, X)

    # Muestro los resultados al usuario final
    st.write("**Predicción de la Huella de Carbono**")
    st.write(f'R-squared (Coeficiente de Determinación): {r2:.2f}')
    st.write(f'Mean Squared Error (Error Cuadrático Medio): {mse:.2f}')
    st.write("**Identificación de Factores Significativos**")
    st.dataframe(feature_importance_df)

    # Grafico la importancia de las características
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='Aquamarine')
    plt.xlabel('Importancia')
    plt.ylabel('Variable')
    plt.title('Importancia de las Variables en la Predicción de Emisiones de Carbono')
    plt.gca().invert_yaxis()
    st.pyplot(plt)

# Pruebas Unitarias
class TestCarbonEmissionModel(unittest.TestCase):

    def setUp(self):
        # Creo un conjunto de datos de ejemplo
        data = {
            'Sex': ['Male', 'Female', 'Female', 'Male'],
            'Diet': ['Meat', 'Vegetarian', 'Vegan', 'Meat'],
            'How Often Shower': [1, 2, 3, 1],
            'Heating Energy Source': ['Gas', 'Electric', 'Gas', 'Electric'],
            'Transport': ['Car', 'Bike', 'Car', 'Bike'],
            'Vehicle Type': ['Sedan', 'None', 'Sedan', 'None'],
            'Vehicle Monthly Distance Km': [1000, 0, 800, 0],
            'Social Activity': [5, 3, 2, 5],
            'Frequency of Traveling by Air': [2, 1, 3, 2],
            'CarbonEmission': [3000, 1500, 2500, 2000]
        }
        self.df = pd.DataFrame(data)
        self.X, self.y = preprocess_data(self.df)

    def test_preprocess_data(self):
        """Prueba que la función preprocess_data retorna X e y correctamente."""
        X, y = preprocess_data(self.df)
        self.assertEqual(X.shape[1], self.df.shape[1] - 1)  
        self.assertEqual(len(y), len(self.df)) 

    def test_train_model(self):
        """Prueba que el modelo se entrena y que el R^2 es mayor que 0."""
        regressor, mse, r2 = train_model(self.X, self.y)
        self.assertTrue(r2 > 0)
        self.assertTrue(mse > 0)

    def test_feature_importance(self):
        """Prueba que la función de importancia de características retorne un DataFrame."""
        regressor, _, _ = train_model(self.X, self.y)
        importance_df = feature_importance(regressor, self.X)
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertTrue('Feature' in importance_df.columns)
        self.assertTrue('Importance' in importance_df.columns)

# Ejecución de las pruebas si se ejecuta directamente este archivo
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)

# Creeo la interfaz de usuario en Streamlit
st.title("Análisis de Huella de Carbono")
st.write("Sube un archivo CSV para analizar las emisiones de carbono.")

# Cargue  del archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

# Ejecutar el análisis si se ha cargado un archivo
if uploaded_file is not None:
    analyze_data(uploaded_file)
