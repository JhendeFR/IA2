import cv2
import os

input_dir = r"C:\Users\jhean\Documents\Universidad\IA2\1er Parcial\ACTAS\imagenes_unificadas"
output_dir = "recortes_votos_presidencia"

x, y, w, h = 500, 600, 800, 1200

os.makedirs(output_dir, exist_ok=True)

extensiones = (".jpg", ".jpeg", ".png", ".tif", ".bmp")
imagenes = [f for f in os.listdir(input_dir) if f.lower().endswith(extensiones)]

print(f"Se encontraron {len(imagenes)} imagenes")

for i, nombre in enumerate(imagenes, 1):
    path_img = os.path.join(input_dir, nombre)
    img = cv2.imread(path_img)
    if img is None:
        print(f"[{i}]No se pudo leer: {nombre}")
        continue

    #Recorte
    mesa_crop = img[y:y+h, x:x+w]

    out_name = f"recorte_{nombre}"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, mesa_crop)

    print(f"[{i}] Recorte guardado en {out_path}")

print("\nProceso finalizado Los recortes estan en:", output_dir)