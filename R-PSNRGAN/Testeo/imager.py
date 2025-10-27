import os
from PIL import Image

"""
imager.py

Funciones para reducir y reescalar imágenes desde código (sin usar la consola).

Funciones principales:
- process_image(input_path, scale, method, output_path=None) -> str
        Reduce la imagen por 'scale' (0 < scale < 1) y la reescala al tamaño original
        usando 'bilinear' o 'bicubic'. Devuelve la ruta del archivo guardado.

- process_images(jobs, output_dir=None) -> list[str]
        Ejecuta process_image para una lista de trabajos. Cada trabajo es un dict con claves:
        {'input': str, 'scale': float, 'method': 'bilinear'|'bicubic', 'output': Optional[str]}

Ejemplo de uso desde código:
        out = process_image("input.jpg", 0.5, "bicubic", "out.jpg")
        # o para múltiples
        jobs = [
                {"input": "a.jpg", "scale": 0.5, "method": "bilinear"},
                {"input": "b.png", "scale": 0.3, "method": "bicubic", "output": "b_res.png"},
        ]
        outs = process_images(jobs, output_dir="results")
"""

_INTERPOLATIONS = {
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
}


def process_image(input_path: str, scale: float, method: str, output_path: str | None = None) -> str:
        """
        Reduce y reescala una imagen. Devuelve la ruta del archivo guardado.
        Lanza ValueError en caso de parámetros inválidos.
        """
        if method not in _INTERPOLATIONS:
                raise ValueError(f"Método desconocido: {method}. Elija 'bilinear' o 'bicubic'.")

        if not (0 < scale < 1):
                raise ValueError("El factor de escala debe estar entre 0 y 1 (reducción).")

        img = Image.open(input_path)
        orig_size = img.size  # (width, height)

        # Tamaño reducido
        reduced_size = (max(1, int(orig_size[0] * scale)), max(1, int(orig_size[1] * scale)))
        reduced = img.resize(reduced_size, resample=Image.LANCZOS)

        # Reescalado al tamaño original usando el método elegido
        resample_filter = _INTERPOLATIONS[method]
        restored = reduced.resize(orig_size, resample=resample_filter)

        # Determinar ruta de salida
        if not output_path:
                base, ext = os.path.splitext(input_path)
                output_path = f"{base}_rescaled_{method}{ext}"

        # Crear carpeta si es necesario
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

        # Guardar resultado
        restored.save(output_path)
        return output_path


def process_images(jobs: list[dict], output_dir: str | None = None) -> list[str]:
        """
        Procesa múltiples trabajos. Cada job es un dict con claves:
        - 'input'  (str) required
        - 'scale'  (float) required
        - 'method' (str) required
        - 'output' (str) optional

        Si se proporciona output_dir, las salidas sin ruta absoluta se colocan allí.
        Devuelve lista de rutas de salida.
        """
        results = []
        for job in jobs:
                inp = job["input"]
                scale = job["scale"]
                method = job["method"]
                out = job.get("output")

                if output_dir and out:
                        # si out es nombre simple, colocarlo en output_dir
                        if not os.path.isabs(out) and os.path.basename(out) == out:
                                out = os.path.join(output_dir, out)
                elif output_dir and not out:
                        base, ext = os.path.splitext(os.path.basename(inp))
                        out = os.path.join(output_dir, f"{base}_rescaled_{method}{ext}")

                results.append(process_image(inp, scale, method, out))
        return results


if __name__ == '__main__':
        # Ruta fija de entrada (archivo dentro del repositorio)
        INPUT_PATH = r"C:\Users\jhean\Documents\Universidad\IA2\R-PSNRGAN\LR\eca9485be474d5ca65558d6e4b88cab8.png"
        # Tamaño objetivo (1080x1080) — el script redimensionará directamente a esta resolución
        TARGET_SIZE = (1080, 1080)
        # Métodos a aplicar: guardaremos dos imágenes reescaladas, una bilinear y otra bicubic
        METHODS = ["bilinear", "bicubic"]

        # Carpeta de salida: colocamos los resultados en LR/rescaled para no mezclar
        OUT_DIR = os.path.join(os.path.dirname(INPUT_PATH), "rescaled")
        os.makedirs(OUT_DIR, exist_ok=True)

        # Abrir la imagen de entrada una vez
        img = Image.open(INPUT_PATH)

        saved = []
        for method in METHODS:
                if method not in _INTERPOLATIONS:
                        print(f"Método desconocido {method}, saltando.")
                        continue
                resample_filter = _INTERPOLATIONS[method]
                # Redimensionar directamente a TARGET_SIZE sin reducción previa
                resized = img.resize(TARGET_SIZE, resample=resample_filter)
                base, ext = os.path.splitext(os.path.basename(INPUT_PATH))
                out_name = f"{base}_1080x1080_{method}{ext}"
                out_path = os.path.join(OUT_DIR, out_name)
                resized.save(out_path)
                saved.append(out_path)

        # Imprimir rutas guardadas
        for p in saved:
                print(p)