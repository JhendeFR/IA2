#Prueba de interpolacion de modelos
import sys
import torch
from collections import OrderedDict

# Lee alpha (factor de mezcla entre 0 y 1) desde los argumentos de la línea de comandos
alpha = float(0.7)
net_PSNR_path = './models/RRDB_PSNR_x4.pth'
net_ESRGAN_path = './models/RRDB_ESRGAN_x4.pth'
# Ruta de salida: guarda el modelo interpolado con sufijo según alpha*10 (dos dígitos)
net_interp_path = './models/interpolacion_{:02d}.pth'.format(int(alpha*10))

# Carga el estado de pesos del modelo PSNR
net_PSNR = torch.load(net_PSNR_path)
# Carga el estado de pesos del modelo ESRGAN
net_ESRGAN = torch.load(net_ESRGAN_path)
# Diccionario ordenado donde se almacenarán los pesos interpolados
net_interp = OrderedDict()

# Mensaje informativo con el alpha usado
print('Interpolating with alpha = ', alpha)

# Recorre cada parámetro del modelo PSNR para mezclarlo con el de ESRGAN
for k, v_PSNR in net_PSNR.items():
    # Obtiene el parámetro correspondiente del modelo ESRGAN (misma clave)
    v_ESRGAN = net_ESRGAN[k]
    # Interpolación lineal de pesos: (1-alpha)*PSNR + alpha*ESRGAN
    net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN

# Guarda el nuevo checkpoint interpolado en disco
torch.save(net_interp, net_interp_path)