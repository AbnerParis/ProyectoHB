import streamlit as st
import base64
from pathlib import Path

def show_local_drawio(drawio_path):
    """
    Muestra un archivo DrawIO local en Streamlit usando el visor de diagrams.net
    """
    try:
        drawio_path = Path(drawio_path)
        if not drawio_path.exists():
            st.error(f"Archivo no encontrado: {drawio_path}")
            return

        # Read as binary to avoid encoding issues
        with open(drawio_path, "rb") as f:  # 'rb' para leer como binario
                    diagram_data = f.read()
                    encoded_xml = base64.b64encode(diagram_data).decode("utf-8")  # Codifica correctamente

        # Encode to base64
        encoded_xml = base64.b64encode(diagram_data).decode()
        
        # Create viewer URL
        viewer_url = f"https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&layers=1&nav=1&xml={encoded_xml}"

        # Display in Streamlit
        st.components.v1.html(
        f'<iframe src="{viewer_url}" width="100%" height="600px" frameborder="0"></iframe>',
        height=600,)

    except Exception as e:
        st.error(f"Error al cargar el diagrama: {str(e)}")
        raise e  # Re-raise the exception for the calling code to handle