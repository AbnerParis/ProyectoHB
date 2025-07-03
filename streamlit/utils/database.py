import csv
import mysql.connector
from mysql.connector import Error
import pandas as pd


config = {
    'user': 'root', # nombre de usuario
    'password': 'admin', # contraseña
    'host': 'localhost',
    'database': 'coches', # nombre de la base de datos
    'raise_on_warnings': True
}

def create_connection():
    """Crear conexión a la base de datos"""
    try:
        conn = mysql.connector.connect(**config)
        return conn
    except Error as e:
        print(f"Error al conectar a MySQL: {e}")
        return None
    
def fetch_data(query):
    conn = create_connection()
    if conn is None:
        return None

    cursor = conn.cursor()
    cursor.execute(query) # Ejecutamos la query
    column_names = cursor.column_names # Nombre de las columnas de la tabla

    # Guardamos los datos de la tabla
    data = cursor.fetchall()

    cursor.close()
    conn.close()
    df = pd.DataFrame(data, columns=column_names)
    
    return df 

fetch_data("""
SELECT 
    v.id_extraccion,
    v.timestamp_extraccion,
    v.marca,
    v.precio_contado AS precio,
    v.financiacion_disponible,
    u.ubicacion,
    v.kilometraje,
    c.combustible,
    v.es_km0,
    v.es_demo,
    u.latitud,
    u.longitud,
    t.transmision,
    tc.tipo_carroceria,
    v.asientos,
    v.potencia_cv AS potencia,
    v.puertas,
    v.mes_matriculacion,
    v.año_matriculacion,
    v.garantia,
    v.modelo
FROM 
    vehiculos v
LEFT JOIN 
    ubicaciones u ON v.ubicacion_id = u.id
LEFT JOIN 
    tipos_carroceria tc ON v.tipo_carroceria_id = tc.id
LEFT JOIN
    combustibles c ON v.combustible_id = c.id
LEFT JOIN
    transmisiones t ON v.transmision_id = t.id
ORDER BY 
    v.created_at DESC
LIMIT 1000;
""")
