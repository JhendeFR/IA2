import os
import sys
import shutil
import hashlib
import pandas as pd
from pathlib import Path

# ================== CONFIG ==================
CSV_NAME = "descargas.csv"              # nombre del CSV (junto al script)
COL_RUTA = "Ruta"                       # columna con la ruta
DEST_DIR = "imagenes_unificadas"        # carpeta destino única
IMG_EXTS = {".jpg",".jpeg",".png",".tif",".tiff",".bmp",".gif"}
# ============================================

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def try_resolve(raw: str, base_csv: Path):
    """
    Devuelve una lista de rutas candidatas (Path) que probaremos en orden.
    raw: texto de la columna Ruta (puede ser relativo con 'ACTAS\\...')
    base_csv: carpeta donde está el CSV (y el script)
    """
    raw = (raw or "").strip().strip('"').strip("'")
    if not raw:
        return []

    # Normalizar separadores
    raw_norm = raw.replace("\\", os.sep).replace("/", os.sep)

    # Variantes a probar
    candidates = []

    p_raw = Path(raw_norm)
    candidates.append(p_raw)  # tal cual (absoluta o relativa al cwd)

    # relativa a la carpeta del CSV
    candidates.append(base_csv / raw_norm)

    # si empieza con 'ACTAS'/'actas', quitar ese prefijo y probar relativo al CSV
    parts = Path(raw_norm).parts
    if parts and parts[0].lower() == "actas":
        without_actas = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        candidates.append(base_csv / without_actas)

        # también relativo al padre de base_csv (por si base_csv ya ES ACTAS)
        candidates.append(base_csv.parent / raw_norm)
        candidates.append(base_csv.parent / without_actas)
    else:
        # también intentar relativo al padre (por si CSV está adentro de ACTAS)
        candidates.append(base_csv.parent / raw_norm)

    # Quitar duplicados preservando orden
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq

def gather_images_from(path: Path):
    """Si path es archivo imagen -> [path].
       Si path es carpeta -> caminar y devolver todas las imágenes dentro.
       Si no existe -> [].
    """
    if path.is_file():
        return [path] if is_image(path) else []
    if path.is_dir():
        imgs = []
        for root, _, files in os.walk(path):
            for fn in files:
                p = Path(root) / fn
                if is_image(p):
                    imgs.append(p)
        return imgs
    return []

def main():
    here = Path(__file__).resolve().parent
    csv_path = here / CSV_NAME
    if not csv_path.exists():
        print(f"❌ No encontré {csv_path}. Coloca el script junto al CSV.")
        sys.exit(1)

    dest = here / DEST_DIR
    dest.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if COL_RUTA not in df.columns:
        print(f"❌ La columna '{COL_RUTA}' no existe. Columnas: {list(df.columns)}")
        sys.exit(1)

    manifest_rows = []
    copiados = 0
    faltantes = 0
    carpetas_vacias = 0

    for idx, row in df.iterrows():
        raw = str(row[COL_RUTA]).strip()
        resolved = None
        imgs = []

        for cand in try_resolve(raw, csv_path.parent):
            if cand.exists():
                resolved = cand
                imgs = gather_images_from(cand)
                break

        if not resolved:
            manifest_rows.append({
                "fila": idx,
                "ruta_original": raw,
                "ruta_resuelta": "",
                "copiado_a": "",
                "sha256": "",
                "estado": "no_encontrada"
            })
            print(f"[{idx}] SIN_IMAGENES: {resolved}")
            faltantes += 1
            continue

        if not imgs:
            # Existe pero no hay imágenes (p.ej. carpeta vacía o archivo no imagen)
            manifest_rows.append({
                "fila": idx,
                "ruta_original": raw,
                "ruta_resuelta": str(resolved),
                "copiado_a": "",
                "sha256": "",
                "estado": "sin_imagenes"
            })
            print(f"[{idx}] NO_ENCONTRADA: {raw}")
            carpetas_vacias += 1
            continue

        for k, img_path in enumerate(imgs):
            # nombre único: index_fila[-k]_nombre
            out_name = f"{idx:06d}"
            if len(imgs) > 1:
                out_name += f"_{k:02d}"
            out_name += "_" + img_path.name
            out_path = dest / out_name

            shutil.copy2(img_path, out_path)
            copiados += 1

            sha = sha256_of(out_path)
            manifest_rows.append({
                "fila": idx,
                "ruta_original": raw,
                "ruta_resuelta": str(resolved),
                "copiado_a": str(out_path),
                "sha256": sha,
                "estado": "copiado"
            })
            print(f"[{idx}] COPIADO: {img_path} -> {out_path}")

    man = pd.DataFrame(manifest_rows)
    man_path = here / "manifest_copiado.csv"
    man.to_csv(man_path, index=False, encoding="utf-8-sig")

    print("\n==== RESUMEN ====")
    print(f"Imágenes copiadas: {copiados}")
    print(f"Rutas no encontradas: {faltantes}")
    print(f"Rutas sin imágenes: {carpetas_vacias}")
    print(f"Destino: {dest}")
    print(f"Manifiesto: {man_path}")

if __name__ == "__main__":
    main()
