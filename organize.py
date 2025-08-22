import os
import shutil
import random

# 📁 Ruta al dataset original
DATASET_DIR = 'C:\\Users\\jhean\\Documents\\Universidad\\IA2\\DataSet P01'  # Ejemplo: './dataset_original'
OUTPUT_DIR = 'C:\\Users\\jhean\\Documents\\Universidad\\IA2\\DataSet P01\\dataset_dividido'  # Ejemplo: './dataset_dividido'

# 📊 Proporciones
SPLIT_RATIOS = {
    'train': 0.80,
    'test': 0.15,
    'val': 0.05
}

# 🛠 Función para crear carpetas destino
def create_dirs(classes):
    for split in SPLIT_RATIOS.keys():
        for cls in classes:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# 🔄 Función para dividir y copiar imágenes
def split_dataset():
    classes = os.listdir(DATASET_DIR)
    create_dirs(classes)

    for cls in classes:
        class_path = os.path.join(DATASET_DIR, cls)
        images = os.listdir(class_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(SPLIT_RATIOS['train'] * total)
        test_end = train_end + int(SPLIT_RATIOS['test'] * total)

        splits = {
            'train': images[:train_end],
            'test': images[train_end:test_end],
            'val': images[test_end:]
        }

        for split, split_images in splits.items():
            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(OUTPUT_DIR, split, cls, img)
                shutil.copy2(src, dst)

    print("✅ División completada con éxito.")

# 🚀 Ejecutar
if __name__ == '__main__':
    split_dataset()