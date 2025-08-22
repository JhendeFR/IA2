import os
from PIL import Image

# 📁 Ruta de la carpeta con las imágenes
carpeta = r"C:\Users\jhean\Documents\Universidad\IA2\Descargas"
nombre_base = 'Mother-Boards_'
extensiones_validas = ['.jpg', '.jpeg', '.png', '.webp']

contador = 1

for archivo in sorted(os.listdir(carpeta)):
    ruta_completa = os.path.join(carpeta, archivo)
    nombre, extension = os.path.splitext(archivo)

    if extension.lower() in extensiones_validas and os.path.isfile(ruta_completa):
        nuevo_nombre = f"{nombre_base}{str(contador).zfill(3)}{extension.lower()}"
        nueva_ruta = os.path.join(carpeta, nuevo_nombre)

        # Redimensionar imagen
        try:
            with Image.open(ruta_completa) as img:
                img = img.resize((800, 600), Image.LANCZOS)
                img.save(nueva_ruta)
            os.remove(ruta_completa)
            print(f"✅ {archivo} → {nuevo_nombre} (redimensionada)")
            contador += 1
        except Exception as e:
            print(f"❌ Error procesando {archivo}: {e}")

print(f"\n🎉 Renombrado y redimensionado completo: {contador - 1} archivos procesados.")