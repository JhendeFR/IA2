import os
import re
import time
import base64
import hashlib
import requests
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Configuración
PALABRA_CLAVE = "Porno tetas grandes"
PREFIX = "Mother-Boards"                          # Etiqueta para el nombre de archivo
CARPETA_DESTINO = r"C:\Users\jhean\Documents\Universidad\IA2\Descargas"
NUMERO_SCROLLS = 15
TAMANIO_OBJETIVO = (800, 600)              # Ancho x Alto

# Crear carpeta si no existe
os.makedirs(CARPETA_DESTINO, exist_ok=True)

# Calcular siguiente índice basándonos en los archivos existentes
patron = re.compile(rf"{re.escape(PREFIX)}_(\d+)\.jpg$")
indices = []
for nombre in os.listdir(CARPETA_DESTINO):
    m = patron.match(nombre)
    if m:
        indices.append(int(m.group(1)))
start_index = max(indices) + 1 if indices else 1

# Inicializar contador
contador = start_index

# Inicializar navegador
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)

# Navegar a Google Imágenes
driver.get("https://www.google.com/imghp")
search_box = driver.find_element(By.NAME, "q")
search_box.send_keys(PALABRA_CLAVE)
search_box.send_keys(Keys.RETURN)

time.sleep(3)  # Esperar carga inicial

# Scroll para cargar más imágenes
for _ in range(NUMERO_SCROLLS):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

imagenes = driver.find_elements(By.TAG_NAME, "img")
print(f"Detectadas {len(imagenes)} imágenes")

hashes_guardados = set()

for img in imagenes:
    src = img.get_attribute("src")
    if not src:
        continue

    try:
        # Cargar imagen (base64 o URL)
        if src.startswith("data:image"):
            header, encoded = src.split(",", 1)
            data = base64.b64decode(encoded)
        else:
            resp = requests.get(src, timeout=5)
            if resp.status_code != 200:
                continue
            data = resp.content

        im = Image.open(BytesIO(data))

        # Filtrar tamaño mínimo
        if im.width < 25 or im.height < 25:
            continue

        # Filtrar contenido mínimo
        if len(im.tobytes()) < 5000:
            continue

        # Evitar duplicados
        img_hash = hashlib.md5(im.tobytes()).hexdigest()
        if img_hash in hashes_guardados:
            continue
        hashes_guardados.add(img_hash)

        # Redimensionar y convertir a RGB
        im = im.convert("RGB")
        im_resized = im.resize(TAMANIO_OBJETIVO, Image.LANCZOS)

        # Guardar con nombre secuencial de 2 dígitos
        nombre_archivo = f"{PREFIX}_{contador:02d}.jpg"
        ruta = os.path.join(CARPETA_DESTINO, nombre_archivo)
        im_resized.save(ruta, "JPEG")
        contador += 1

    except Exception:
        continue

print(f"Guardadas {contador - start_index} imágenes en '{CARPETA_DESTINO}'")
driver.quit()