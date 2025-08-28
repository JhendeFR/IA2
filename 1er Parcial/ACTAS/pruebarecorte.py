import cv2
import os

# ================== CONFIG ==================
img_path = r"C:\Users\jhean\Documents\Universidad\IA2\1er Parcial\ACTAS\imagenes_unificadas\000001_mesa_3_8004251.jpg"   # ruta a tu imagen original del acta
output_dir = "Mesas"                     # carpeta de salida
output_name = "votos.jpg"         # nombre del archivo recortado
# ============================================

# Crear carpeta destino
os.makedirs(output_dir, exist_ok=True)

# Leer imagen
img = cv2.imread(img_path)

# --- Coordenadas de recorte ---
# Como punto de partida: define manualmente (x, y, w, h)
# Estos valores funcionan para tu acta de ejemplo (ajústalos según necesites)
# En la imagen de 3336x2200 el bloque "MESA" está aprox. en la esquina izquierda.
x, y, w, h = 500, 600, 800, 1200   # <--- AJUSTA ESTOS VALORES

# Recortar
mesa_crop = img[y:y+h, x:x+w]

# Guardar
out_path = os.path.join(output_dir, output_name)
cv2.imwrite(out_path, mesa_crop)

print(f"Recorte guardado en {out_path}")
