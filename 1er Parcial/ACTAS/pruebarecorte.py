import cv2
import os

img_path = r"C:\Users\jhean\Documents\Universidad\IA2\1er Parcial\ACTAS\imagenes_unificadas\000001_mesa_3_8004251.jpg"
output_dir = "Mesas"
output_name = "votos.jpg"

os.makedirs(output_dir, exist_ok=True)

#Leer la imagen
img = cv2.imread(img_path)
#Configuracion de recorte
x, y, w, h = 500, 600, 800, 1200 

#Recortar
mesa_crop = img[y:y+h, x:x+w]

# Guardar
out_path = os.path.join(output_dir, output_name)
cv2.imwrite(out_path, mesa_crop)

print(f"Recorte guardado en {out_path}")
