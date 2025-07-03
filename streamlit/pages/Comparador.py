import streamlit as st
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
import os
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.database import fetch_data


df = fetch_data("""
        SELECT v.*, t.transmision AS transmision
        FROM vehiculos v
        LEFT JOIN transmisiones t ON v.transmision_id = t.id;
        """)
#eliminar outliers de precio_contado
df = df[df['precio_contado'] < 1000000]  # Ajusta
with st.expander("🔍 COMPARADOR DE VEHÍCULOS", expanded=True):
    st.header("Comparador de Vehículos")
    st.markdown("Selecciona dos vehículos para comparar sus características principales.")
    
    # Limpieza y conversión de tipos de datos
    df_clean = df.copy()
    
    # Convertir columnas numéricas (asegurando que sean numéricas)
    numeric_cols = {
        'precio_contado': float,
        'kilometraje': float,
        'potencia_cv': float,
        'puertas': int,
        'asientos': int,
        'año_matriculacion': int
    }
    
    for col, dtype in numeric_cols.items():
        if col in df_clean.columns:
            # Convertir a string primero para manejar casos especiales
            df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(',', '.'), errors='coerce')
            # Rellenar NaN con la mediana
            df_clean[col] = df_clean[col].fillna(df_clean[col].median()).astype(dtype)
    
    # Calcular antigüedad
    df_clean['antiguedad'] = 2025 - df_clean['año_matriculacion']
    
    # Obtener lista única de vehículos (marca + modelo)
    df_clean['marca_modelo'] = df_clean['marca'] + ' ' + df_clean['modelo']
    vehiculos_unicos = df_clean['marca_modelo'].unique()
    
    # Selectores para elegir los vehículos a comparar
    col1, col2 = st.columns(2)
    with col1:
        vehiculo1 = st.selectbox(
            "Selecciona el primer vehículo:",
            options=vehiculos_unicos,
            index=0,
            key="vehiculo1"
        )
    with col2:
        vehiculo2 = st.selectbox(
            "Selecciona el segundo vehículo:",
            options=vehiculos_unicos,
            index=1 if len(vehiculos_unicos) > 1 else 0,
            key="vehiculo2"
        )
    
    # Filtrar datos para los vehículos seleccionados
    try:
        datos_v1 = df_clean[df_clean['marca_modelo'] == vehiculo1].iloc[0]
        datos_v2 = df_clean[df_clean['marca_modelo'] == vehiculo2].iloc[0]
    except IndexError:
        st.error("No se encontraron datos completos para los vehículos seleccionados.")
        st.stop()
    
    # Seleccionar características a comparar
    features = [
        ('precio_contado', 'Precio (€)', 'currency'),
        ('kilometraje', 'Kilometraje (km)', 'number'),
        ('antiguedad', 'Antigüedad (años)', 'number'),
        ('potencia_cv', 'Potencia (CV)', 'number'),
        ('puertas', 'Número de puertas', 'number'),
        ('asientos', 'Número de asientos', 'number'),
        ('transmision', 'Transmisión', 'text')
    ]
    
    # Mostrar información básica
    st.subheader("Comparando:")
    st.write(f"**Vehículo 1:** {vehiculo1}")    
    st.write(f"**Vehículo 2:** {vehiculo2}")
    
    # Gráfico de radar - solo con características numéricas
    # Gráfico de radar - solo con características numéricas
    fig_comparacion = go.Figure()

    # Obtener máximos para normalización (excluyendo transmision)
    numeric_features = [f for f in features if f[2] != 'text']
    max_values = {col: df_clean[col].max() for col, _, _ in numeric_features if col in df_clean.columns}
    min_values = {col: df_clean[col].min() for col, _, _ in numeric_features if col in df_clean.columns}

    for vehiculo, datos, color in [(vehiculo1, datos_v1, '#1f77b4'),
                                   (vehiculo2, datos_v2, '#ff7f0e')]:
        valores = []
        categorias = []
        hover_text = []
        
        for col, nombre, tipo in numeric_features:
            if col in datos and pd.notna(datos[col]):
                valor = float(datos[col])  # Asegurar que es float
                
                # Normalización diferente según la característica
                if col == 'kilometraje':
                    # Mantenemos la normalización invertida original para kilometraje
                    valor_norm = (1 - (valor / max_values[col])) * 100
                else:
                    # Normalización min-max estándar (0-100) para precio y otras características
                    valor_norm = ((valor - min_values[col]) / (max_values[col] - min_values[col])) * 100
                
                if tipo == 'currency':
                    display_val = f"{valor:,.0f} €"
                else:
                    display_val = f"{valor:,.0f}"
                
                valores.append(valor_norm)
                categorias.append(nombre)
                hover_text.append(f"{nombre}: {display_val}")
        
        fig_comparacion.add_trace(go.Scatterpolar(
            r=valores,
            theta=categorias,
            fill='toself',
            name=vehiculo,
            line=dict(color=color),
            hoverinfo='text',
            hovertext=hover_text
        ))
    # Configurar el gráfico
    fig_comparacion.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        margin=dict(t=80, b=20, l=20, r=20),
        

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5
        )
    )
    
    st.plotly_chart(fig_comparacion, use_container_width=True)
    
    # Mostrar tabla comparativa detallada
    st.subheader("Comparación Detallada")
    
    # Crear tabla de comparación
    comparacion_data = []
    for col, nombre, tipo in features:
        if col in datos_v1 and col in datos_v2 and pd.notna(datos_v1[col]) and pd.notna(datos_v2[col]):
            try:
                if tipo == 'currency':
                    valor1 = float(datos_v1[col])
                    valor2 = float(datos_v2[col])
                    display1 = f"{valor1:,.0f} €"
                    display2 = f"{valor2:,.0f} €"
                    diff = ((valor1 - valor2)/valor2*100) if valor2 != 0 else 0
                elif tipo == 'number':
                    valor1 = float(datos_v1[col])
                    valor2 = float(datos_v2[col])
                    display1 = f"{valor1:,.0f}"
                    display2 = f"{valor2:,.0f}"
                    diff = ((valor1 - valor2)/valor2*100) if valor2 != 0 else 0
                else:  # texto
                    display1 = str(datos_v1[col])
                    display2 = str(datos_v2[col])
                    diff = "-"
                
                comparacion_data.append({
                    "Característica": nombre,
                    vehiculo1: display1,
                    vehiculo2: display2,
                    "Diferencia": f"{diff:+.1f}%" if diff != "-" else diff
                })
            except (ValueError, TypeError):
                continue
    
    # Mostrar tabla
    if comparacion_data:
        st.dataframe(
            pd.DataFrame(comparacion_data),
            hide_index=True,
            column_config={
                "Característica": st.column_config.TextColumn(width="medium"),
                vehiculo1: st.column_config.TextColumn(width="small"),
                vehiculo2: st.column_config.TextColumn(width="small"),
                "Diferencia": st.column_config.TextColumn(
                    "Diferencia %",
                    help="Porcentaje de diferencia (Vehículo 1 respecto a Vehículo 2)"
                )
            }
        )
    else:
        st.warning("No hay datos suficientes para mostrar la comparación.")