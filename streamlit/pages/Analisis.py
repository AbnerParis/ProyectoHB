import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from utils.database import fetch_data
from dotenv import load_dotenv
import os           
from decimal import Decimal
import numpy as np


# Configuración de la página
st.set_page_config(page_title="Análisis de Vehículos", layout="wide")

# Título principal
st.title("Análisis de Mercado de Vehículos")

# Función principal
def main():
    # Sección de introducción
    st.markdown("""
    
    Esta aplicación muestra diferentes perspectivas sobre los datos de vehículos, incluyendo:
    - Detección de valores atípicos en precios
    - Relación entre precio y antigüedad
    - Relación entre kilometraje y precio
    - Distribución por marcas
    - Precio vs kilometraje y tipo de transmisión
    - Mapa de coches en venta por ubicacion
    """)

    # Corrige la definición de tabs (faltaba una coma y el nombre de la sexta pestaña)
    tab1, tab2, tab3, tab4, tab5, tab6= st.tabs([
        "Outliers en Precios", 
        "Precio vs Antigüedad",
        "Kilometraje vs Precio",
        "Análisis por Marca",
        "Precio vs Kilometraje y Tipo de Transmisión",
        "Mapa de Vehículos"
    ])


    df = fetch_data("""
        SELECT v.*, t.transmision AS transmision
        FROM vehiculos v
        LEFT JOIN transmisiones t ON v.transmision_id = t.id;
        """)
    
    
    # Pestaña 1: Outliers en precios
    with tab1:
        st.header("Detección de Outliers en Precios")
        st.markdown("""
        Identificación de valores atípicos en los precios de vehículos utilizando el método de Tukey's Fence.
        Los outliers pueden indicar precios anormalmente altos o bajos en el mercado.
        """)
        
        # Asegurar que los valores de precios sean decimales
        df['precio_contado'] = df['precio_contado'].apply(lambda x: float(str(x)))
        
        # Normalización logarítmica para mejor visualización
        df['precio_log'] = np.log10(df['precio_contado'] + 1)  # +1 para evitar log(0)
        
        # Función para detectar outliers con Tukey's Fence
        def detect_outliers_tukey(data, column):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            return outliers, lower_bound, upper_bound

        # Detectar outliers en datos normalizados
        outliers_log, lower_bound_log, upper_bound_log = detect_outliers_tukey(df, 'precio_log')
        df['es_outlier'] = (df['precio_log'] < lower_bound_log) | (df['precio_log'] > upper_bound_log)
        
        # Crear figura interactiva con subplots
        fig = make_subplots(rows=2, cols=1, 
                            subplot_titles=("Distribución Original de Precios", 
                                        "Distribución Normalizada (log10)"),
                            vertical_spacing=0.15)
        
        # Gráfico 1: Distribución original con outliers
        fig.add_trace(
            go.Histogram(
                x=df[df['es_outlier'] == False]['precio_contado'],
                name='Precios Normales',
                marker_color='#1f77b4',
                opacity=0.7,
                nbinsx=50
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=df[df['es_outlier'] == True]['precio_contado'],
                name='Outliers',
                marker_color='#ff7f0e',
                opacity=0.7,
                nbinsx=50
            ),
            row=1, col=1
        )
        
        # Gráfico 2: Distribución normalizada
        fig.add_trace(
            go.Histogram(
                x=df[df['es_outlier'] == False]['precio_log'],
                name='Precios Normales',
                marker_color='#1f77b4',
                opacity=0.7,
                nbinsx=50,
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=df[df['es_outlier'] == True]['precio_log'],
                name='Outliers',
                marker_color='#ff7f0e',
                opacity=0.7,
                nbinsx=50,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Añadir líneas de límites al gráfico normalizado
        fig.add_vline(
            x=lower_bound_log, 
            line_dash="dash", 
            line_color="green", 
            annotation_text=f"Límite inferior: {10**lower_bound_log:.2f}€",
            annotation_position="bottom right",
            row=2, col=1
        )

        fig.add_vline(
            x=upper_bound_log, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"Límite superior: {10**upper_bound_log:.2f}€",
            annotation_position="top right",
            row=2, col=1
        )
        
        # Personalizar el layout
        fig.update_layout(
            title_text='<b>Análisis de Outliers con y sin Normalización</b>',
            title_x=0.5,
            height=800,
            width=900,
            bargap=0.1,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Actualizar ejes
        fig.update_xaxes(title_text="Precio (€)", row=1, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
        fig.update_xaxes(title_text="log10(Precio)", row=2, col=1)
        fig.update_yaxes(title_text="Frecuencia", row=2, col=1)
        
        # Mostrar la figura en Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar información sobre los outliers
        st.subheader("Resumen de Outliers")
        
        # Convertir los límites logarítmicos de vuelta a escala original
        lower_bound_original = 10**lower_bound_log
        upper_bound_original = 10**upper_bound_log
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Límite inferior", f"{lower_bound_original:.2f}€")
            st.metric("Número de outliers bajos", len(df[df['precio_contado'] < lower_bound_original]))
        
        with col2:
            st.metric("Límite superior", f"{upper_bound_original:.2f}€")
            st.metric("Número de outliers altos", len(df[df['precio_contado'] > upper_bound_original]))
        
        # Mostrar outliers por marca en un gráfico de barras
        st.subheader("Outliers por Marca")
        marcas_outliers = outliers_log['marca'].value_counts().reset_index()
        marcas_outliers.columns = ['Marca', 'Número de Outliers']
        
        fig_marcas = px.bar(
            marcas_outliers, 
            x='Marca', 
            y='Número de Outliers',
            color='Número de Outliers',
            text='Número de Outliers',
            title='<b>Distribución de Outliers por Marca</b>'
        )
        fig_marcas.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_marcas, use_container_width=True)
        
        # Mostrar ejemplos de outliers
        st.subheader("Ejemplos de Outliers")
        outliers_original = df[df['es_outlier'] == True].sort_values('precio_contado', ascending=False)
        st.dataframe(
            outliers_original[['marca', 'modelo', 'precio_contado']].head(10),
            column_config={
                "precio_contado": st.column_config.NumberColumn(
                    "Precio (€)",
                    format="%.2f €"
                )
            }
        )
        
    # Pestaña 2: Precio vs Antigüedad
    with tab2:
    
    
        st.header("Relación entre Precio y Antigüedad - Top 7 Marcas")
        st.markdown("""
        Análisis de cómo la antigüedad afecta el precio para las 7 marcas principales.
        """)
        
        # Calcular antigüedad
        df["antiguedad"] = 2025 - df["año_matriculacion"]
        
        # Identificar las 5 marcas principales
        top_marcas = df['marca'].value_counts().head(7).index.tolist()
        df_top = df[df['marca'].isin(top_marcas)]
        
        # Crear pestañas para diferentes visualizaciones
        tab2_1, tab2_2 = st.tabs(["Distribución por Antigüedad", "Tendencia Promedio"])
        
        with tab2_1:
            # Gráfico de barras agrupado por antigüedad
            fig_barras = px.bar(
                df_top,
                x="antiguedad",
                y="precio_contado",
                color="marca",
                barmode="group",
                title="Distribución de Precios por Antigüedad (Top 7 Marcas)",
                labels={"antiguedad": "Antigüedad (años)", "precio_contado": "Precio Promedio (€)"},
                hover_data=["modelo", "kilometraje"],
                height=600
            )
            
            # Mejorar formato
            fig_barras.update_layout(
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_barras, use_container_width=True)
        
        with tab2_2:
            # Calcular precios promedio por antigüedad y marca
            df_avg = df_top.groupby(['marca', 'antiguedad'], as_index=False).agg({
                'precio_contado': 'mean',
                'kilometraje': 'mean'
            })
            
            # Gráfico de líneas de tendencia
            fig_line = px.line(
                df_avg,
                x="antiguedad",
                y="precio_contado",
                color="marca",
                title="Tendencia Promedio de Precio por Antigüedad (Top 7 Marcas)",
                labels={"antiguedad": "Antigüedad (años)", "precio_contado": "Precio Promedio (€)"},
                markers=True,
                height=600,
                hover_data=["kilometraje"]
            )
            
            # Personalizar hover
            fig_line.update_traces(
                hovertemplate="<b>%{fullData.name}</b><br>Antigüedad: %{x} años<br>Precio promedio: %{y:.2f}€<br>Km promedio: %{customdata[0]:,.0f}<extra></extra>"
            )
            
            fig_line.update_layout(
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Análisis de depreciación (se mantiene igual)
        st.subheader("Análisis Comparativo de Depreciación")
        
        # Calcular tasa de depreciación
        depreciacion = []
        for marca in top_marcas:
            marca_df = df_top[df_top['marca'] == marca]
            precio_nuevo = marca_df[marca_df['antiguedad'] <= 1]['precio_contado'].median()
            precio_5años = marca_df[(marca_df['antiguedad'] >= 4) & (marca_df['antiguedad'] <= 6)]['precio_contado'].median()
            tasa = (precio_nuevo - precio_5años) / precio_nuevo * 100 if precio_nuevo > 0 else 0
            depreciacion.append({"Marca": marca, "Depreciación a 5 años (%)": round(tasa, 1)})
        
        # Mostrar métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mayor depreciación", 
                    max(depreciacion, key=lambda x: x["Depreciación a 5 años (%)"])["Marca"],
                    f"{max(depreciacion, key=lambda x: x['Depreciación a 5 años (%)'])['Depreciación a 5 años (%)']}%")
        
        with col2:
            st.metric("Menor depreciación", 
                    min(depreciacion, key=lambda x: x["Depreciación a 5 años (%)"])["Marca"],
                    f"{min(depreciacion, key=lambda x: x['Depreciación a 5 años (%)'])['Depreciación a 5 años (%)']}%")
        
        with col3:
            avg_deprec = sum(d["Depreciación a 5 años (%)"] for d in depreciacion) / len(depreciacion)
            st.metric("Depreciación promedio", f"{avg_deprec:.1f}%")
        
        # Gráfico de barras horizontal para depreciación
        fig_deprec = px.bar(
            pd.DataFrame(depreciacion).sort_values("Depreciación a 5 años (%)", ascending=True),
            x="Depreciación a 5 años (%)",
            y="Marca",
            orientation='h',
            title="Depreciación a 5 años por Marca",
            color="Depreciación a 5 años (%)",
            color_continuous_scale='RdYlGn_r',  # Rojo (alta depreciación) a Verde (baja depreciación)
            height=400
        )
        
        fig_deprec.update_layout(
            yaxis={'categoryorder':'total ascending'},
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig_deprec, use_container_width=True)
        
        st.markdown("""
        **Observaciones clave:**
        - Las marcas premium suelen mostrar menor depreciación
        - La mayor caída de valor ocurre en los primeros 3 años
        - El kilometraje promedio aumenta proporcionalmente con la antigüedad
        - Algunas marcas mantienen mejor su valor en el largo plazo
        """)
    
    
    with tab3:
        
        st.header("Relación entre Kilometraje y Precio (Top 7 Marcas)")
        st.markdown("""
        Análisis de cómo el kilometraje afecta el precio para las 7 marcas principales (limitado a 500,000 km).
        """)

        # Filtrar por kilometraje máximo de 500,000 km
        df_filtered = df[df['kilometraje'] <= 500000]

        # Identificar las 7 marcas principales
        top_7_marcas = df_filtered['marca'].value_counts().head(7).index.tolist()
        df_top7 = df_filtered[df_filtered['marca'].isin(top_7_marcas)].copy()

        # Calcular pendientes (depreciación) para ordenar las marcas
        pendientes = {}
        for marca in top_7_marcas:
            marca_df = df_top7[df_top7['marca'] == marca]
            if len(marca_df) > 1:
                slope, _ = np.polyfit(marca_df['kilometraje'], marca_df['precio_contado'], 1)
                pendientes[marca] = slope

        # Ordenar marcas por depreciación (de mayor a menor pendiente negativa)
        marcas_ordenadas = sorted(pendientes.keys(), key=lambda x: pendientes[x])

        # Normalizar precios solo para estas marcas
        scaler = MinMaxScaler()
        df_top7['precio_normalizado'] = scaler.fit_transform(df_top7[['precio_contado']])

        # Crear bins de kilometraje cada 100,000 km
        df_top7['kilometraje_bin'] = pd.cut(
            df_top7['kilometraje'],
            bins=[0, 100000, 200000, 300000, 400000, 500000],
            precision=0
        ).astype(str)

        # Pestaña para el gráfico
        tab3 = st.tabs(["Tendencia por Marca"])[0]

        with tab3:
            # Agrupar por bins de kilometraje
            df_grouped = df_top7.groupby(['marca', 'kilometraje_bin'], as_index=False).agg({
                'precio_normalizado': ['mean', 'std'],
                'kilometraje': 'mean',
                'precio_contado': 'median'
            })
            
            # Aplanar columnas multi-index
            df_grouped.columns = ['_'.join(col).strip('_') for col in df_grouped.columns.values]
            
            # Gráfico de líneas con puntos
            fig = go.Figure()
            
            # Usar el orden calculado por depreciación
            for marca in marcas_ordenadas:
                marca_df = df_grouped[df_grouped['marca'] == marca].sort_values('kilometraje_mean')
                
                # Línea principal
                fig.add_trace(go.Scatter(
                    x=marca_df['kilometraje_mean'],
                    y=marca_df['precio_normalizado_mean'],
                    name=marca,
                    mode='lines+markers',
                    marker=dict(size=8),
                    line=dict(width=2),
                    hovertemplate="<b>%{fullData.name}</b><br>Kilometraje: %{x:,.0f} km<br>Precio normalizado: %{y:.3f}<br>Precio mediano: €%{customdata[0]:,.0f}<extra></extra>",
                    customdata=marca_df['precio_contado_median'].values.reshape(-1,1)
                ))
            
            # Añadir líneas verticales cada 100,000 km
            for km in [100000, 200000, 300000, 400000, 500000]:
                fig.add_vline(
                    x=km, 
                    line=dict(color="lightgray", width=1, dash="dot"),
                    annotation_text=f"{km//1000}k km",
                    annotation_position="top"
                )
            
            fig.update_layout(
                title="Tendencia Precio/Kilometraje (Top 7 Marcas) - Puntos cada 100k km",
                xaxis_title="Kilometraje (km)",
                yaxis_title="Precio Normalizado (0-1)",
                height=600,
                hovermode="x unified",
                plot_bgcolor='rgba(240,240,240,0.8)',
                xaxis=dict(
                    range=[0, 500000],
                    tickvals=[0, 100000, 200000, 300000, 400000, 500000],
                    ticktext=["0", "100k", "200k", "300k", "400k", "500k"]
                ),
                legend_title=f"Marcas (ordenadas por depreciación)<br>↓ Mayor a menor ↓"
            )
            
            st.plotly_chart(fig, use_container_width=True)

        # Análisis cuantitativo (se mantiene igual)
        st.subheader("Análisis Comparativo")

        # Calcular métricas de correlación y pendiente
        correlaciones = []
        for marca in top_7_marcas:
            marca_df = df_top7[df_top7['marca'] == marca]
            
            # Calcular correlación
            corr = marca_df['kilometraje'].corr(marca_df['precio_contado'])
            
            # Calcular regresión lineal
            slope, intercept = np.polyfit(marca_df['kilometraje'], marca_df['precio_contado'], 1)
            
            correlaciones.append({
                "Marca": marca,
                "Correlación": corr,
                "Impacto": "Fuerte" if abs(corr) > 0.5 else "Moderado" if abs(corr) > 0.3 else "Débil",
                "Pendiente": slope,  # €/km
                "Pérdida_por_10k_km": f"€{abs(slope)*10000:,.0f}"
            })

        # Mostrar métricas en 3 columnas
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Mayor correlación negativa", 
                    max(correlaciones, key=lambda x: -x["Correlación"])["Marca"],
                    f"{max(correlaciones, key=lambda x: -x['Correlación'])['Correlación']:.2f}")

        with col2:
            st.metric("Menor correlación negativa", 
                    min(correlaciones, key=lambda x: x["Correlación"])["Marca"],
                    f"{min(correlaciones, key=lambda x: x['Correlación'])['Correlación']:.2f}")

        with col3:
            max_slope = max(correlaciones, key=lambda x: abs(x["Pendiente"]))
            st.metric("Mayor depreciación (por 10k km)", 
                    max_slope["Marca"],
                    max_slope["Pérdida_por_10k_km"])

        # Gráfico de correlaciones
        df_corr = pd.DataFrame(correlaciones).sort_values("Correlación")

        fig_corr = px.bar(
            df_corr,
            x="Correlación",
            y="Marca",
            color="Correlación",
            color_continuous_scale='RdYlGn_r',
            orientation='h',
            title="Correlación Kilometraje-Precio por Marca",
            height=400,
            hover_data=["Pérdida_por_10k_km"]
        )

        fig_corr.update_layout(
            yaxis={'categoryorder':'total ascending'},
            coloraxis_showscale=False,
            xaxis_range=[-1, 0.1]
        )

        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("""
        **Interpretación:**
        - **Correlaciones negativas:** Valores cercanos a -1 indican fuerte relación inversa entre kilometraje y precio
        - **Depreciación:** Muestra cuánto valor pierde cada marca por cada 10,000 km recorridos
        - **Datos limitados a 500,000 km:** Análisis enfocado en vehículos con menos de medio millón de kilómetros
        """)
        
    # Pestaña 4: Análisis por marca
    with tab4:
    
        st.header("Análisis por Marca (Top 10)")
        st.markdown("""
        Comparación de precios promedio y distribución para las 10 marcas principales.
        """)
        
        # Identificar las 10 marcas principales
        top_10_marcas = df['marca'].value_counts().head(10).index.tolist()
        df_top10 = df[df['marca'].isin(top_10_marcas)]
        
        # Ordenar marcas por precio promedio (de mayor a menor)
        marca_order = df_top10.groupby('marca')['precio_contado'].mean().sort_values(ascending=False).index
        
        # Crear pestañas para diferentes visualizaciones
        tab4_1, tab4_2 = st.tabs(["Precios Promedio", "Distribución Completa"])
        
        with tab4_1:
            # Gráfico de barras de precios promedio ordenado
            fig4a = px.bar(
                df_top10.groupby('marca', as_index=False)['precio_contado'].mean().sort_values('precio_contado', ascending=False),
                x='marca',
                y='precio_contado',
                title='<b>Precio Promedio por Marca (Top 10)</b>',
                labels={'marca': 'Marca', 'precio_contado': 'Precio Promedio (€)'},
                color='marca',
                category_orders={"marca": marca_order},
                text_auto='.2s',
                height=500
            )
            
            # Mejorar formato
            fig4a.update_layout(
                showlegend=False,
                xaxis_tickangle=-45,
                hovermode="x unified",
                yaxis_tickprefix="€",
                yaxis_tickformat=",."
            )
            
            fig4a.update_traces(
                textfont_size=12,
                textangle=0,
                textposition="outside",
                cliponaxis=False
            )
            
            st.plotly_chart(fig4a, use_container_width=True)
            
            # Mostrar tabla con datos
            st.subheader("Datos Detallados")
            st.dataframe(
                df_top10.groupby('marca')['precio_contado'].agg(['mean', 'median', 'count', 'min', 'max'])
                .sort_values('mean', ascending=False)
                .rename(columns={
                    'mean': 'Precio Promedio',
                    'median': 'Precio Mediano',
                    'count': 'Vehículos',
                    'min': 'Precio Mínimo',
                    'max': 'Precio Máximo'
                }),
                column_config={
                    "Precio Promedio": st.column_config.NumberColumn(format="%.2f €"),
                    "Precio Mediano": st.column_config.NumberColumn(format="%.2f €")
                }
            )
        
        with tab4_2:
            #with tab4_2:
    # Filtrar datos para precios <= 150000
            df_top10_filtered = df_top10[df_top10['precio_contado'] <= 150000]
            
            # Boxplot ordenado por precio promedio
            fig4b = px.box(
                df_top10_filtered,  # Usar el DataFrame filtrado
                x='marca',
                y='precio_contado',
                title='<b>Distribución de Precios por Marca (Top 10) - Hasta €150,000</b>',
                labels={'marca': 'Marca', 'precio_contado': 'Precio (€)'},
                color='marca',
                category_orders={"marca": marca_order},
                height=600
            )
            
            # Mejorar formato
            fig4b.update_layout(
                showlegend=False,
                xaxis_tickangle=-45,
                hovermode="x unified",
                yaxis_tickprefix="€",
                yaxis_tickformat=",.",
                yaxis_range=[0, 150000]  # Asegurar que el eje Y no muestre valores superiores
            )
            
            st.plotly_chart(fig4b, use_container_width=True)
            
        
        # Análisis comparativo
        st.subheader("Análisis Comparativo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_price = df_top10['precio_contado'].mean()
            st.metric("Precio promedio total", f"{avg_price:,.2f} €")
        
        with col2:
            max_brand = df_top10.groupby('marca')['precio_contado'].mean().idxmax()
            max_price = df_top10.groupby('marca')['precio_contado'].mean().max()
            st.metric("Marca más premium", max_brand, f"{max_price:,.2f} €")
        
        with col3:
            min_brand = df_top10.groupby('marca')['precio_contado'].mean().idxmin()
            min_price = df_top10.groupby('marca')['precio_contado'].mean().min()
            st.metric("Marca más accesible", min_brand, f"{min_price:,.2f} €")
        
        st.markdown("""
        **Insights clave:**
        - Las marcas se ordenan de mayor a menor precio promedio
        - El boxplot muestra la dispersión de precios para cada marca
        - El gráfico de violín revela la densidad de precios
        - Algunas marcas tienen rangos de precios más amplios que otras
        """)
    
    
        #Pestaña 5: Precio vs Kilometraje y Tipo de Transmisión
        
    with tab5:
            st.header("Análisis de Precio vs Kilometraje y Tipo de Transmisión")
            st.markdown("""
            Exploración de cómo el tipo de transmisión afecta la relación entre precio y kilometraje.
            """)

            # Filtrar datos con transmisión válida y sin nulos en precio_contado y kilometraje
            df_transmision = df.dropna(subset=['transmision', 'precio_contado', 'kilometraje'])
            
            # Filtrar solo transmisiones Automático y Manual (si existen otros valores)
            df_transmision = df_transmision[df_transmision['transmision'].isin(['Automático', 'Manual'])]

            # Definir colores personalizados para las transmisiones
            color_map = {'Manual': '#1f77b4', 'Automático': '#ff7f0e'}
            
            # Crear gráfico de dispersión
            fig6 = px.scatter(
                df_transmision,
                x="kilometraje",
                y="precio_contado",
                color="transmision",
                color_discrete_map=color_map,  # Aplicar el mapeo de colores
                opacity=0.4,
                title='Precio vs Kilometraje por tipo de transmisión',
                labels={
                    "kilometraje": "Kilometraje (log)",
                    "precio_contado": "Precio (€, log)",
                    "transmision": "Transmisión"
                },
                hover_data={
                    "marca": True,
                    "modelo": True,
                    "kilometraje": ":.0f",
                    "precio_contado": ":.0f"
                },
                log_x=True,
                log_y=True
            )

            fig6.update_layout(
                plot_bgcolor='white',
                xaxis=dict(
                    gridcolor='lightgray',
                    gridwidth=0.6,
                    showgrid=True,
                    linecolor='black',
                    title="Kilometraje (escala logarítmica)"
                ),
                yaxis=dict(
                    gridcolor='lightgray',
                    gridwidth=0.5,
                    showgrid=True,
                    linecolor='black',
                    title="Precio (€, escala logarítmica)"
                ),
                legend_title_text='Transmisión',
                legend=dict(
                    x=1.02,  # Mover ligeramente fuera del gráfico
                    y=1,
                    xanchor='left',
                    yanchor='top'
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode='closest'
            )
            
            # Mejorar el formato del hover
            fig6.update_traces(
                hovertemplate="<br>".join([
                    "<b>%{customdata[0]} %{customdata[1]}</b>",
                    "Transmisión: %{fullData.name}",
                    "Kilometraje: %{x:,.0f} km",
                    "Precio: %{y:,.0f} €"
                ])
            )

            st.plotly_chart(fig6, use_container_width=True)
                    
        #pestaña 6: Mapa cloropético de coches en venta por ubicación
    with tab6:  
        st.header("Mapa de Densidad de Vehículos por Ubicación")
        st.markdown("""
        Visualización de la concentración geográfica de vehículos en venta.
        """)

        # Verificar si tenemos datos de geolocalización
        if 'latitud' in df.columns and 'longitud' in df.columns:
            # Filtrar datos con coordenadas válidas
            df_geo = df.dropna(subset=['latitud', 'longitud'])
            df_geo = df_geo[(df_geo['latitud'] != 0) & (df_geo['longitud'] != 0)]
            
            # Contar vehículos por ubicación para la densidad
            # Primero redondeamos las coordenadas para agrupar cercanías
            df_geo['lat_rounded'] = df_geo['latitud'].round(2)
            df_geo['lon_rounded'] = df_geo['longitud'].round(2)
            
            # Contamos vehículos por ubicación redondeada
            density_data = df_geo.groupby(['lat_rounded', 'lon_rounded']).size().reset_index(name='count')
            
            # Crear mapa de densidad
            fig7 = px.density_mapbox(
                density_data,
                lat="lat_rounded",
                lon="lon_rounded",
                z="count",  # Usamos el conteo de vehículos en lugar del precio
                radius=20,  # Aumentamos el radio para mejor visualización
                center=dict(lat=df_geo['latitud'].mean(), lon=df_geo['longitud'].mean()),
                zoom=5,
                mapbox_style="open-street-map",
                title="<b>Concentración Geográfica de Vehículos en Venta</b>",
                color_continuous_scale=px.colors.sequential.Plasma,
                labels={"count": "N° de vehículos"},
                hover_data={"lat_rounded": False, "lon_rounded": False, "count": True}
            )
            
            # Mejorar el diseño
            fig7.update_layout(
                margin=dict(l=0, r=0, t=40, b=0),
                coloraxis_colorbar=dict(
                    title="N° vehículos",
                    thicknessmode="pixels",
                    thickness=15,
                    lenmode="pixels",
                    len=300,
                    yanchor="top",
                    y=1,
                    ticks="outside"
                )
            )
            
            # Personalizar el hover
            fig7.update_traces(
                hovertemplate="<b>Cantidad de vehículos:</b> %{z}<extra></extra>"
            )
            
            st.plotly_chart(fig7, use_container_width=True)
            
            # Mostrar estadísticas
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Vehículos con ubicación válida", len(df_geo))
            with col2:
                st.metric("Zonas con mayor concentración", 
                        f"{density_data['count'].max()} vehículos",
                        f"en {density_data.loc[density_data['count'].idxmax(), 'lat_rounded']:.2f}°, {density_data.loc[density_data['count'].idxmax(), 'lon_rounded']:.2f}°")
            
            # Mostrar tabla de las 10 zonas con más vehículos
            st.subheader("Top 10 Zonas con Mayor Concentración")
            st.dataframe(
                density_data.sort_values('count', ascending=False).head(10).reset_index(drop=True),
                column_config={
                    "lat_rounded": st.column_config.NumberColumn("Latitud", format="%.2f°"),
                    "lon_rounded": st.column_config.NumberColumn("Longitud", format="%.2f°"),
                    "count": st.column_config.NumberColumn("Vehículos")
                },
                hide_index=True
            )
            
        else:
            st.warning("No se encontraron datos de ubicación geográfica válidos.")      

if __name__ == "__main__":
    main()
