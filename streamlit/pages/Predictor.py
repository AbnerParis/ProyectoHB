import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

def get_metric_value(metrics_dict, key, default=0):
    value = metrics_dict.get(key, default)
    if isinstance(value, (list, np.ndarray)):
        return value[0] if len(value) > 0 else default
    return value

# Configuración de la página
st.set_page_config(page_title="Predictor de Precios", layout="wide")

# Cargar y limpiar dataframe
try:
    df_2 = pd.read_csv('C:\\Users\\abner\\proyecto\\Proyecto-\\DB\\datos_limpios.csv')
    df_2 = df_2.dropna(subset=['combustible', 'tipo_carroceria', 'transmision'])

    df_d = df_2['modelo'].unique().tolist()
    df_g = df_2['combustible'].unique().tolist()
    df_c = df_2['tipo_carroceria'].unique().tolist()
    df_t = df_2['transmision'].unique().tolist()

except Exception as e:
    st.error(f"Error al cargar los datos: {str(e)}")
    st.stop()

@st.cache_resource
def load_models_and_metrics():
    try:
        with open('C:\\Users\\abner\\proyecto\\Proyecto-\\streamlit\\mejor_modelo_top.pkl', 'rb') as f:
            ml_model = pickle.load(f)

        dl_model = keras.models.load_model('C:\\Users\\abner\\proyecto\\Proyecto-\\streamlit\\modelo_red_neuronal_mae.keras')

        with open("C:\\Users\\abner\\proyecto\\Proyecto-\\streamlit\\ohe_combustible.pkl", "rb") as f:
            ohe_combustible = pickle.load(f)

        with open("C:\\Users\\abner\\proyecto\\Proyecto-\\streamlit\\target_encoding_modelo.pkl", "rb") as f:
            target_encoding_modelo = pickle.load(f)

        with open("C:\\Users\\abner\\proyecto\\Proyecto-\\streamlit\\ohe_tipo_carroceria.pkl", "rb") as f:
            ohe_tipo_carroceria = pickle.load(f)

        with open("C:\\Users\\abner\\proyecto\\Proyecto-\\streamlit\\standard_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("C:\\Users\\abner\\proyecto\\Proyecto-\\streamlit\\metricas_modelo_ml.pkl", "rb") as f:
            metricas_ml = pickle.load(f)

        with open("C:\\Users\\abner\\proyecto\\Proyecto-\\streamlit\\metricas_modelo_dl.pkl", "rb") as f:
            metricas_dl = pickle.load(f)

        return ml_model, dl_model, ohe_combustible, target_encoding_modelo, ohe_tipo_carroceria, scaler, metricas_ml, metricas_dl

    except Exception as e:
        st.error(f"Error al cargar los modelos o métricas: {str(e)}")
        return None, None, None, None, None, None, None, None

ml_model, dl_model, ohe_combustible, target_encoding_modelo, ohe_tipo_carroceria, scaler, metricas_ml, metricas_dl = load_models_and_metrics()

# Mostrar especificaciones del modelo
if ml_model and dl_model and metricas_ml and metricas_dl:
    st.sidebar.subheader("Especificaciones del Modelo")
    st.sidebar.write("""
    **Modelos disponibles:**
    1. Machine Learning (Regresion Lineal)
    2. Deep Learning (Red Neuronal)

    **Features requeridos:**
    - Modelo
    - Kilometraje
    - Potencia (CV)
    - Año de matriculación
    - Combustible
    - Tipo de carrocería
    - Transmisión
    - Financiación disponible
    """)

# Crear pestañas
tab1, tab2 = st.tabs(["Predicción", "📊 Feature Importance"])

with tab1:
    st.title("🔧 Predictor de Precios de Vehículos (ML + DL)")

    with st.form("vehicle_form"):
        st.header("Ingrese los detalles del vehículo")
        col1, col2 = st.columns(2)

        with col1:
            modelo = st.selectbox("Modelo del Vehículo", options=df_d)
            kilometraje = st.number_input("Kilometraje (km)", min_value=0, value=50000)
            potencia_cv = st.number_input("Potencia (CV)", min_value=50, max_value=2000, value=120)
            año_matriculacion = st.slider("Año de matriculación", 2000, 2025, 2020)

        with col2:
            combustible = st.selectbox("Tipo de combustible", options=df_g)
            tipo_carroceria = st.selectbox("Tipo de carrocería", options=df_c)
            transmision = st.radio("Tipo de transmisión", options=df_t, index=0)
            financiacion_disponible = st.checkbox("Financiación disponible")

        submitted = st.form_submit_button("Predecir Precio")

    if submitted:
        try:
            input_usuario = pd.DataFrame({
                'modelo': [modelo],
                'kilometraje': [kilometraje],
                'potencia_cv': [potencia_cv],
                'año_matriculacion': [año_matriculacion],
                'combustible': [combustible],
                'tipo_carroceria': [tipo_carroceria],
                'transmision': [transmision],
                'financiacion_disponible': [financiacion_disponible]
            })

            input_usuario['modelo_te'] = input_usuario['modelo'].map(target_encoding_modelo)

            combustible_encoded = ohe_combustible.transform(input_usuario[['combustible']])
            combustible_cols = [f"combustible_{cat}" for cat in ohe_combustible.categories_[0]]
            df_combustible = pd.DataFrame(combustible_encoded, columns=combustible_cols, index=input_usuario.index)
            input_usuario = pd.concat([input_usuario, df_combustible], axis=1)
            input_usuario.drop(columns=['combustible'], inplace=True)

            tipo_carroceria_encoded = ohe_tipo_carroceria.transform(input_usuario[['tipo_carroceria']])
            tipo_carroceria_cols = [f"tipo_carroceria_{cat}" for cat in ohe_tipo_carroceria.categories_[0]]
            df_tipo_carroceria = pd.DataFrame(tipo_carroceria_encoded, columns=tipo_carroceria_cols, index=input_usuario.index)
            input_usuario = pd.concat([input_usuario, df_tipo_carroceria], axis=1)
            input_usuario.drop(columns=['tipo_carroceria'], inplace=True)

            input_usuario['transmision_bin'] = input_usuario['transmision'].apply(lambda x: 1 if x == 'Automática' else 0).astype(int)
            input_usuario['financiacion_disponible'] = input_usuario['financiacion_disponible'].astype(int)

            column_order = [
                'modelo_te', 'kilometraje', 'potencia_cv', 'año_matriculacion',
                'combustible_Diésel', 'tipo_carroceria_Deportivo', 'transmision_bin',
                'combustible_Eléctrico', 'financiacion_disponible',
                'combustible_Híbrido Enchufable'
            ]
            for col in column_order:
                if col not in input_usuario.columns:
                    input_usuario[col] = 0
            input_usuario = input_usuario[column_order]

            input_data = scaler.transform(input_usuario)

            precio_ml = ml_model.predict(input_data)[0]
            precio_dl = dl_model.predict(input_data)[0][0]

            st.subheader("🔍 Resultados de la Predicción")
            col1, col2 = st.columns(2)

            with col1:
                st.success(f"## 🖥️ ML (Regresion Lineal)")
                st.success(f"### Precio estimado: €{precio_ml:,.2f}")
                st.write("**Métricas de evaluación:**")
                st.write(f"- MAE: {get_metric_value(metricas_ml, 'MAE'):,.2f}")
                st.write(f"- MSE: {get_metric_value(metricas_ml, 'MSE'):,.2f}")
                st.write(f"- RMSE: {get_metric_value(metricas_ml, 'RMSE'):,.2f}")
                st.write(f"- R²: {get_metric_value(metricas_ml, 'R2'):.3f}")

            with col2:
                st.info(f"## 🧠 DL (Red Neuronal)")
                st.info(f"### Precio estimado: €{precio_dl:,.2f}")
                st.write("**Métricas de evaluación:**")
                st.write(f"- Loss: {get_metric_value(metricas_dl, 'loss'):.4f}")
                st.write(f"- MSE: {get_metric_value(metricas_dl, 'mse'):.4f}")
                st.write(f"- Val Loss: {get_metric_value(metricas_dl, 'val_loss'):.4f}")
                st.write(f"- Val MSE: {get_metric_value(metricas_dl, 'val_mse'):.4f}")

            diferencia = abs(precio_ml - precio_dl)
            st.warning(f"**Diferencia entre modelos:** €{diferencia:,.2f} ({diferencia/min(precio_ml, precio_dl)*100:.1f}%)")

        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")


with tab2:
    st.header("📊 Documentación de Feature Importance")
    
    st.markdown("""
    ## ¿Qué es Feature Importance?
    
    La importancia de características es una métrica que nos ayuda a entender:
    - Qué variables influyen más en las predicciones
    - Cómo contribuye cada característica al resultado
    - Qué características podríamos eliminar sin afectar el rendimiento
    """)
    
    st.markdown("""
    ## Métodos de Cálculo
    
    ### Para modelos de árboles:
    - Basado en reducción promedio de impureza (Gini/entropía)
    - Frecuencia de uso para dividir nodos
    
    ### Para modelos lineales:
    - Basado en coeficientes (valor absoluto)
    - Requiere datos normalizados
    
    ### Para redes neuronales:
    - Permutation Importance
    - SHAP values
    """)
    
    st.markdown("""
    ## Interpretación
    
    1. Valores altos = características más importantes
    2. Valores bajos = poco impacto
    3. Importancia relativa > valores absolutos
    4. No implica causalidad
    """)
    
    if ml_model is not None and hasattr(ml_model, 'feature_importances_'):
        st.markdown("""
        ## Feature Importance en tu Modelo
        
        Tu modelo calcula la importancia basándose en:
        - Reducción de impureza en los árboles
        - Frecuencia de uso en divisiones
        
        Recomendaciones:
        - Eliminar características con importancia cercana a cero
        - Analizar las más importantes para entender el modelo
        """)
        
        try:
            features_show = [
                ('Modelo (codificado)', 'modelo_te'),
                ('Kilometraje', 'kilometraje'),
                ('Potencia (CV)', 'potencia_cv'),
                ('Año matriculación', 'año_matriculacion'),
                ('Combustible Diésel', 'combustible_Diésel'),
                ('Tipo carrocería Deportivo', 'tipo_carroceria_Deportivo'),
                ('Transmisión Automática', 'transmision_bin'),
                ('Combustible Eléctrico', 'combustible_Eléctrico'),
                ('Financiación disponible', 'financiacion_disponible'),
                ('Combustible Híbrido Enchufable', 'combustible_Híbrido Enchufable')
            ]
            
            importance_df = pd.DataFrame({
                'Característica': [x[0] for x in features_show],
                'Importancia': ml_model.feature_importances_
            }).sort_values('Importancia', ascending=False)
            
            st.subheader("Importancia de Características")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(importance_df['Característica'], importance_df['Importancia'])
                ax.set_xlabel('Importancia')
                ax.set_title('Importancia Relativa de Características')
                st.pyplot(fig)
            
            with col2:
                st.dataframe(importance_df.style.format({'Importancia': '{:.4f}'}))
                
        except Exception as e:
            st.warning(f"No se pudo mostrar la importancia: {str(e)}")

# Instrucciones y métricas en el sidebar
st.sidebar.markdown(f"""
### 📝 Instrucciones:
1. Complete todos los campos del formulario
2. Haga clic en **"Predecir Precio"**
3. Compare los resultados de ambos modelos

### 📊 Métricas de Referencia:
**ML (Regresion Lineal):**
- MAE: {get_metric_value(metricas_ml, 'MAE'):,.2f}
- MSE: {get_metric_value(metricas_ml, 'MSE'):,.2f}
- R²: {get_metric_value(metricas_ml, 'R2'):.3f}

**DL (Red Neuronal):**
- Loss: {get_metric_value(metricas_dl, 'loss'):.4f}
- Val Loss: {get_metric_value(metricas_dl, 'val_loss'):.4f}
""")