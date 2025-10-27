import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import PSNRNet_arch as arch
from collections import OrderedDict

# Configuración
model_path = 'models/RRDB_PSNR_x4.pth'
test_img_folder = r'LR/*'
out_dir = 'resultado'
output_size = (1080, 1080)  # Resolución deseada (ancho, alto)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Usando:', device)

model = arch.PSNRNet(3, 3, 64, 23, gc=32)

print('Cargando modelo desde', model_path)
checkpoint = torch.load(model_path, map_location=device)
if isinstance(checkpoint, dict):
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k[len('module.'):] if k.startswith('module.') else k
    new_state_dict[new_key] = v

try:
    model.load_state_dict(new_state_dict, strict=True)
except RuntimeError as e:
    print('Strict load_state_dict failed:', e)
    print('Retrying with strict=False')
    model.load_state_dict(new_state_dict, strict=False)

model.eval()
model = model.to(device)

os.makedirs(out_dir, exist_ok=True)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print('Fallo para leer imagen:', path)
        continue

    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    # 🔧 Redimensionar a 1080p
    output_resized = cv2.resize(output, output_size, interpolation=cv2.INTER_CUBIC)

    out_path = osp.join(out_dir, f'{base}_1080p.png')
    cv2.imwrite(out_path, output_resized)
    print('Guardado', out_path)