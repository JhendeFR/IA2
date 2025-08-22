import os, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import cv2

# -------- CONFIG --------
MODEL_TYPE   = "scratch"  # "scratch" o "vgg19"
MODEL_PATH   = "best_cnn_from_scratch.pt"  # o "best_cnn_from_scratch.pt"
DATASET_DIR  = "dataset_dividido_v1"
IMG_PATH     = r"C:\Users\jhean\Downloads\Cloro-Clorox-Anti-Splash-Botella-1900-g.webp" 
# 👆 cambia esta ruta a la imagen que quieras probar
IMG_SIZE     = 224
TOPK         = 3
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Leer clases ====
with open(os.path.join(DATASET_DIR, "classes.txt"), "r", encoding="utf-8") as f:
    CLASSES = [line.strip() for line in f if line.strip()]
NUM_CLASSES = len(CLASSES)

# ==== Definiciones de modelo ====
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        chs = [3,32,64,128,256]
        self.stage1 = nn.Sequential(ConvBlock(chs[0],chs[1]), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(ConvBlock(chs[1],chs[2]), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(ConvBlock(chs[2],chs[3]), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(ConvBlock(chs[3],chs[4]), nn.MaxPool2d(2))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(chs[4], num_classes)
    def forward(self,x):
        x=self.stage1(x); x=self.stage2(x); x=self.stage3(x); x=self.stage4(x)
        x=self.gap(x).flatten(1); x=self.dropout(x); return self.fc(x)

def build_model(model_type:str, num_classes:int):
    if model_type=="scratch":
        return SimpleCNN(num_classes)
    elif model_type=="vgg19":
        weights = models.VGG19_Weights.IMAGENET1K_V1
        model = models.vgg19(weights=weights)
        model.classifier = nn.Sequential(
            nn.Linear(25088,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024,num_classes)
        )
        return model
    else:
        raise ValueError("model_type debe ser 'scratch' o 'vgg19'")

# ==== Cargar modelo ====
model = build_model(MODEL_TYPE, NUM_CLASSES).to(device)
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
print(f"Modelo cargado: {MODEL_TYPE} | Pesos: {MODEL_PATH}")

# ==== Transformaciones ====
tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ==== Funciones de predicción ====
def predict_image(img_path):
    from matplotlib import pyplot as plt
    img = Image.open(img_path).convert("RGB")
    x = tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    idxs = probs.argsort()[::-1][:TOPK]
    topk = [(CLASSES[i], float(probs[i])) for i in idxs]

    print(f"\nImagen: {img_path}")
    for cls,p in topk:
        print(f"  {cls}: {p*100:.1f}%")

    plt.imshow(img); plt.axis("off")
    plt.title("Predicción principal: "+topk[0][0])
    plt.show()

def predict_webcam(cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = tfms(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        top_idx = int(probs.argmax())
        label, prob = CLASSES[top_idx], probs[top_idx]
        cv2.putText(frame,f"{label}:{prob*100:.1f}%",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("Webcam (q para salir)", frame)
        if cv2.waitKey(1)&0xFF==ord("q"): break
    cap.release(); cv2.destroyAllWindows()

# ==== Pregunta interactiva ====
print("\n=== MODO INFERENCIA ===")
print("1 = Usar la imagen definida en IMG_PATH")
print("2 = Usar cámara")
choice = input("Elige opción (1/2): ").strip()

if choice=="1":
    predict_image(IMG_PATH)
elif choice=="2":
    print("Iniciando webcam... (q para salir)")
    predict_webcam(0)
else:
    print("Opción inválida")
