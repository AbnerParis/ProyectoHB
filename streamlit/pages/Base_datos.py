import streamlit as st
from pathlib import Path

def main():
    # Configuración de página con estilo moderno en grises
    st.set_page_config(
        page_title="Visualizador de Diagrama de BD",
        layout="wide",
        page_icon="🗄️"
    )
    
    # Estilo elegante en escala de grises
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .css-18e3th9 {
            background-color: #e0e0e0;
        }
        h1 {
            color: #1976D2;
            text-align: center;
            font-family: 'Arial', sans-serif;
            margin-bottom: 30px;
        }
        .stAlert {
            background-color: #eeeeee !important;
        }
        .stMarkdown {
            color: #333333;
        }
        .css-1v3fvcr {
            padding: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🗄️ Diagrama Base de Datos")

    # Ruta al archivo HTML
    html_path = Path("C:\\Users\\abner\\proyecto\\Proyecto-\\streamlit\\diagrama_db2.html")
    
    # Contenedor principal minimalista
    with st.container():
        if html_path.exists():
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Contenedor para el diagrama con borde sutil
                st.markdown(
                    """
                    <div style="background: white; border-radius: 8px; 
                                padding: 10px; box-shadow: 0 2px 4px 0 rgba(0,0,0,0.1);
                                margin: 0 auto; max-width: 1200px;">
                    """,
                    unsafe_allow_html=True
                )
                
                st.components.v1.html(html_content, height=800, scrolling=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error al cargar el diagrama: {str(e)}")
                st.markdown(
                    f"""
                    <div style="margin-top: 20px; text-align: center;">
                        <a href="{html_path}" target="_blank" style="
                            background: #1976D2;
                            color: white;
                            padding: 10px 24px;
                            text-decoration: none;
                            display: inline-block;
                            font-size: 16px;
                            border-radius: 4px;
                            transition: all 0.3s;">
                            Abrir diagrama en nueva pestaña
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.error("Archivo no encontrado: diagramabd.html")
            st.markdown(
                """
                <div style="text-align: center; margin-top: 20px; color: #666;">
                    Por favor, verifica la ruta del archivo HTML
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()