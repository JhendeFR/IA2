#Solo para reducirla escala de imagenes xd
import os
from PIL import Image
from flask import Flask, send_from_directory

# Configuración
INPUT_FOLDER = "Testeo/imageneshd"
OUTPUT_FOLDER = "Testeo/imageneslow"
TARGET_SIZE = (125, 120)

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Procesar imágenes
def procesar_imagenes():
    for nombre_archivo in os.listdir(INPUT_FOLDER):
        ruta_entrada = os.path.join(INPUT_FOLDER, nombre_archivo)
        if os.path.isfile(ruta_entrada):
            try:
                with Image.open(ruta_entrada) as img:
                    img = img.convert('RGB')  # Asegura formato compatible
                    img = img.resize(TARGET_SIZE, Image.LANCZOS)
                    nombre_salida = os.path.splitext(nombre_archivo)[0] + '.png'
                    ruta_salida = os.path.join(OUTPUT_FOLDER, nombre_salida)
                    img.save(ruta_salida, format='PNG')
                    print(f'Procesada: {nombre_archivo} → {nombre_salida}')
            except Exception as e:
                print(f'Error con {nombre_archivo}: {e}')

# Servidor Flask para visualizar imágenes procesadas
app = Flask(__name__)

@app.route('/imagen/<nombre>')
def servir_imagen(nombre):
    return send_from_directory(OUTPUT_FOLDER, nombre)

if __name__ == '__main__':
    procesar_imagenes()
    print("Servidor iniciado en http://localhost:5000/imagen/<nombre>")
    app.run(debug=True)