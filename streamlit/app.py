import streamlit as st

st.set_page_config(
    page_title="Analizador Experto de Coches Usados",
    page_icon="🚗",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?auto=format&fit=crop&w=1500&q=80&grayscale');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .big-font {
        font-size:2.8em !important;
        font-weight: bold;
        color: #ffffff;
        text-shadow: 2px 2px 8px #000000;
    }
    .subtitle {
        font-size:1.4em !important;
        color: #f5f5f5;
        text-shadow: 2px 2px 6px #000000;
    }
    .info-box {
        background: rgba(40,40,40,0.92);
        border-radius: 20px;
        padding: 1.5em;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        margin-bottom: 2em;
        color: white;
        border-left: 5px solid #e53935;
    }
    .desc-box {
        background: rgba(50,50,50,0.93);
        border-radius: 10px;
        padding: 1.5em;
        margin-bottom: 1.5em;
        font-size: 1.1em;
        color: #f5f5f5;
        border-left: 5px solid #1e88e5;
    }
    .stMarkdown ul li {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="big-font">🚗 ANALIZADOR PROFESIONAL DE COCHES USADOS</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Herramienta experta para comprar con confianza | Datos reales del mercado español | Análisis avanzados</div>',
    unsafe_allow_html=True
)

# Descripción general del proyecto y secciones de la app
st.markdown(
    """
    <div class="desc-box">
        <b style="font-size:1.2em;">PODER DE ANÁLISIS PARA TU PRÓXIMA COMPRA</b><br><br>
        Esta plataforma profesional te ofrece <b>herramientas avanzadas</b> para dominar el mercado de coches usados en España. Analiza miles de vehículos con criterios expertos y toma decisiones basadas en datos reales.<br><br>
        <b>VENTAJAS EXCLUSIVAS:</b>
        <ul>
            <li><b>Comparador:</b> Herramienta profesional pero de uso basico, para comparar caracteristicas de un vehiculo</li>
            <li><b>Análisis comparativo:</b> Graficos detallados entre modelos, marcas y años</li>
            <li><b>Indicadores de mercado:</b> Evolución de precios, ratios de depreciación </li>
            <li><b>Predictor:</b> Algoritmo que identifica el precio de tu vehiculo</li>
        </ul>
        <b style="color:#4fc3f7;">Utiliza el menú lateral para acceder a las herramientas profesionales</b>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">
        <h3 style="color:#ff7043; margin-top:0;">¿POR QUÉ USAR ESTA PLATAFORMA?</h3>
        <ul>
            <li>📈 <b>DATOS EN TIEMPO REAL:</b> Accede a la información más actualizada del mercado</li>
            <li>🔎 <b>DETECCIÓN DE GANGAS:</b> Nuestro algoritmo identifica vehículos con mejor relación calidad-precio</li>
            <li>⚙️ <b>HERRAMIENTAS PROFESIONALES:</b> Análisis técnicos que usan los expertos del sector</li>
            <li>💰 <b>AHORRO GARANTIZADO:</b> Usuarios ahorran en promedio un 15-20% en sus compras</li>
        </ul>
        <br>
        <b style="color:#69f0ae;">Selecciona una herramienta en el menú lateral para comenzar tu análisis experto</b>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">', unsafe_allow_html=True)