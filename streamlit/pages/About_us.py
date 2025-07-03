import streamlit as st
from PIL import Image  # Para manejar imágenes

# Configuración de la página
st.set_page_config(
    page_title="About Us | Nuestro Equipo",
    page_icon="👥",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .team-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        margin: 10px;
        transition: 0.3s;
        background-color: #f9f9f9;
    }
    .team-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .social-links a {
        margin-right: 15px;
        color: #2c3e50 !important;
        text-decoration: none;
    }
    .social-links a:hover {
        color: #1abc9c !important;
    }
    .profile-img {
        border-radius: 50%;
        width: 150px;
        height: 150px;
        object-fit: cover;
        margin-bottom: 15px;
        border: 3px solid #1abc9c;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("👥 Nuestro Equipo")
st.markdown("---")

# Contenedor principal
with st.container():
    # Fila 1 - Título
    st.header("Conoce al equipo detrás del proyecto")
    st.write("Somos un grupo apasionado por los datos y el desarrollo de soluciones innovadoras.")
    
    # Fila 2 - Miembros del equipo
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="team-card">
            <center>
                <img src="https://avatars.githubusercontent.com/u/194435398?v=4" class="profile-img">
                <h3>Abner Paris</h3>
                <p><i>Data Scientist</i></p>
                <div class="social-links">
                    <a href="https://www.linkedin.com/in/abner-paris-urunaga" target="_blank">
                        <i class="fab fa-linkedin"></i> LinkedIn
                    </a>
                    <a href="https://github.com/AbnerParis" target="_blank">
                        <i class="fab fa-github"></i> GitHub
                    </a>
                </div>
            </center>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="team-card">
            <center>
                <img src="https://avatars.githubusercontent.com/u/212022578?v=4" class="profile-img">
                <h3>Cristian Torralbo</h3>
                <p><i>Data Scientist</i></p>
                <div class="social-links">
                    <a href="www.linkedin.com/in/cristian-tm-b191b6245" target="_blank">
                        <i class="fab fa-linkedin"></i> LinkedIn
                    </a>
                    <a href="https://github.com/CristianT10" target="_blank">
                        <i class="fab fa-github"></i> GitHub
                    </a>
                </div>
            </center>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="team-card">
            <center>
                <img src="https://avatars.githubusercontent.com/u/34394008?v=4" class="profile-img">
                <h3>Irina Gorria</h3>
                <p><i>Data Scientist</i></p>
                <div class="social-links">
                    <a href="linkedin.com/in/irinagorria" target="_blank">
                        <i class="fab fa-linkedin"></i> LinkedIn
                    </a>
                    <a href="https://github.com/Anirih" target="_blank">
                        <i class="fab fa-github"></i> GitHub
                    </a>
                </div>
            </center>
        </div>
        """, unsafe_allow_html=True)

# Sección de tecnologías
st.markdown("---")
st.header("🛠 Tecnologías que utilizamos")
tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown("""
    - **Python**
    - **Streamlit**
    - **Pandas**
    """)

with tech_col2:
    st.markdown("""
    - **Scikit-learn**
    - **TensorFlow**
    - **Plotly**
    """)

with tech_col3:
    st.markdown("""
    - **Machine Learning**
    - **Git**
    - **MySql**
    """)

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray;'>
    © 2023 Nuestro Equipo | Todos los derechos reservados
</p>
""", unsafe_allow_html=True)