import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# Función para entrenar el modelo de regresión
def train_economic_impact_model(df, rendimiento):
    X = df[['Rendimiento_Cultivo_MT_por_HA']]  # Variable predictora
    y = df['Impacto_Económico_Millones_USD']  # Variable objetivo

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Entrenar modelo de regresión
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Predicciones
    predictions = lin_reg.predict(X_test)
    predictions = lin_reg.predict(rendimiento)

    return lin_reg, predictions

# Función para entrenar el modelo de Random Forest para estrategias adaptativas
def train_strategy_model(df, temp, precipitaciones, salud_suelo, riego, pesticidas, fertilizantes):
    df_n = df[['Temperatura_Promedio_C', 'Precipitacion_Total_mm', 'Acceso_a_Riego_%',
               'Uso_de_Pesticidas_KG_por_HA', 'Uso_de_Fertilizantes_KG_por_HA', 'Índice_de_Salud_Suelo',
               'Estrategias_de_Adaptación']].copy()

    # Mapeo de las estrategias
    estrategia_mapping = {
        'No Adaptation': 0,
        'Water Management': 1,
        'Drought-resistant Crops': 2,
        'Organic Farming': 3,
        'Crop Rotation': 4
    }

    X = df_n.drop(columns=['Estrategias_de_Adaptación'])
    y = df_n['Estrategias_de_Adaptación']

    # Entrenamiento con Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # Predicciones
    input_data = np.array([[temp, precipitaciones, riego, pesticidas, fertilizantes, salud_suelo]])
    estrategia_predicha = rf_model.predict(input_data)
    estrategia_predicha_str = [estrategia_predicha[0]]

    return estrategia_predicha_str

def train_crops_model(df, temp2, precipitaciones2):
    X = df[['Temperatura_Promedio_C', 'Precipitacion_Total_mm']]
    y = df['Tipo_de_Cultivo']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predicciones
    input_data = np.array([[temp2, precipitaciones2]])
    cultivo_predicho = rf_model.predict(input_data)
    cultivo_predicho_str = [cultivo_predicho[0]]

    return cultivo_predicho_str

# Función para cargar los datos directamente desde una URL
@st.cache_data
def load_data():
    return pd.read_csv('impacto_del_cambio_climatico_en_la_agricultura.csv')

# Cargar los datos
df = load_data()

# INTERFAZ
traduccion_estrategia = {
    'No Adaptation': 'Sin adaptación',
    'Water Management': 'Gestión del agua',
    'Drought-resistant Crops': 'Cultivos resistentes a la sequía',
    'Organic Farming': 'Agricultura orgánica',
    'Crop Rotation': 'Rotación de cultivos'
}

traduccion_cultivo = {
    'Wheat': 'Trigo',
    'Corn': 'Maíz',
    'Soybeans': 'Soja',
    'Rice': 'Arroz',
    'Barley': 'Cebada',
    'Sugarcane': 'Caña de azúcar',
    'Cotton': 'Algodón',
    'Vegetables': 'Vegetales',
    'Fruit': 'Frutas',
    'Coffe': 'Café'
}

st.title("Análisis del cambio climático en la agricultura argentina")

st.subheader("¿Qué desea hacer esta vez?")
with st.expander("📈 Predicción del impacto económico en función del rendimiento de los cultivos"):
    rendimiento = st.slider("Rendimiento del cultivo (en toneladas por hectárea)", 0.0, 5.0, 2.5, 0.1)
    rendimiento_reshaped = np.array([[rendimiento]])
    lin_reg, impacto_predicho = train_economic_impact_model(df, rendimiento_reshaped)
    st.write(f"Impacto económico estimado: {impacto_predicho[0]:,.2f} millones de dólares")

    # Gráfico de predicción
    plt.figure(figsize=(8, 6))
    rendimiento_range = np.linspace(0, 5, 100).reshape(-1, 1)
    impacto_range = lin_reg.predict(rendimiento_range)
    plt.scatter(rendimiento, impacto_predicho, color='blue', label='Predicción', alpha=0.7)
    plt.plot(rendimiento_range, impacto_range, color='grey', linestyle='--', label='Línea de predicción')
    plt.xlabel("Rendimiento del cultivo (toneladas/ha)")
    plt.ylabel("Impacto Económico (millones de dólares)")
    plt.title("Impacto económico vs. Rendimiento del cultivo")
    plt.legend()
    st.pyplot(plt)

with st.expander("🌱 Clasificación de las estrategias adaptativas dependiendo de la salud del suelo"):
    temp = st.slider("Temperatura anual promedio (°C)", -5.0, 35.0, 15.0, 0.1, key='slider_temp_estrategias')
    precipitaciones = st.slider("Precipitaciones anuales totales (mm)", 200.0, 3000.0, 1600.0, 0.1, key='slider-precipitaciones_estrategias')
    salud_suelo = st.slider("Índice de salud del suelo", 30.0, 100.0, 65.0, 0.1)
    riego = st.slider("Acceso a riego (%)", 10.0, 100.0, 55.0, 0.1)
    pesticidas = st.slider("Uso de pesticidas (kg/ha)", 0.0, 50.0, 25.0, 0.1)
    fertilizantes = st.slider("Uso de fertilizantes (kg/ha)", 0.0, 100.0, 50.0, 0.1)

    estrategia = train_strategy_model(df, temp, precipitaciones, salud_suelo, riego, pesticidas, fertilizantes)
    estrategia_traducida = traduccion_estrategia.get(estrategia[0])
    st.write(f"La mejor estrategia es: {estrategia_traducida}")

with st.expander("🌾 Clasificación de los cultivos dependiendo del clima"):
    temp2 = st.slider("Temperatura anual promedio (°C)", -5.0, 35.0, 15.0, 0.1, key='slider_temp_cultivos')
    precipitaciones2 = st.slider("Precipitaciones anuales totales (mm)", 200.0, 3000.0, 1600.0, 0.1, key='slider_precipitaciones_cultivos')

    cultivo = train_crops_model(df, temp2, precipitaciones2)
    cultivo_traducido = traduccion_cultivo.get(cultivo[0])
    st.write(f"El cultivo predominante es: {cultivo_traducido}")
