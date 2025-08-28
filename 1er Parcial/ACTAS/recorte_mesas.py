import cv2
import os

# ================== CONFIG ==================
input_dir = r"C:\Users\jhean\Documents\Universidad\IA2\1er Parcial\ACTAS\imagenes_unificadas"
output_dir = "recortes_votos_presidencia"

# Coordenadas del recorte (ajustadas por ti)
x, y, w, h = 500, 600, 800, 1200
# ============================================

# Crear carpeta destino si no existe
os.makedirs(output_dir, exist_ok=True)

# Listar todas las imágenes en la carpeta de entrada
extensiones = (".jpg", ".jpeg", ".png", ".tif", ".bmp")
imagenes = [f for f in os.listdir(input_dir) if f.lower().endswith(extensiones)]

print(f"Se encontraron {len(imagenes)} imágenes para procesar.")

for i, nombre in enumerate(imagenes, 1):
    path_img = os.path.join(input_dir, nombre)
    img = cv2.imread(path_img)
    if img is None:
        print(f"[{i}]No se pudo leer: {nombre}")
        continue

    # Recortar
    mesa_crop = img[y:y+h, x:x+w]

    # Guardar con mismo nombre
    out_name = f"recorte_{nombre}"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, mesa_crop)

    print(f"[{i}] Recorte guardado en {out_path}")

print("\nProceso terminado. Los recortes estan en:", output_dir)