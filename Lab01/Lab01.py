import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class_mapping_path = r"C:\\Users\\jhean\\Documentos\\Tareas\\Inteligencia artificial2\\MLP\\tiny-imagenet-10\\class_mapping.txt"
words_path = r"C:\\Users\\jhean\\Documentos\\Tareas\\Inteligencia artificial2\\MLP\\tiny-imagenet-10\\words.txt"
train_dir = r"C:\\Users\\jhean\\Documentos\\Tareas\\Inteligencia artificial2\\MLP\\tiny-imagenet-10\\train"
test_dir = r"C:\\Users\\jhean\\Documentos\\Tareas\\Inteligencia artificial2\\MLP\\tiny-imagenet-10\\test"

index_to_code = {}
with open(class_mapping_path, "r") as f:
    for line in f:
        index, code = line.strip().split(": ")[0], line.strip().split(": ")[1].split(" - ")[0]
        index_to_code[int(index)] = code

class_descriptions = {}
with open(words_path, "r") as f:
    for line in f:
        code, description = line.strip().split("\t")
        class_descriptions[code] = description

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
testset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform_test)

train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

class MLPClassifier(nn.Module):
    def __init__(self, input_size=64*64*3, hidden_size=512, num_classes=10):
        super(MLPClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MLPClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

def train_model(model, loader, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(loader, desc=f"Época {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1} - Pérdida promedio: {running_loss/len(loader):.4f}")

def evaluate_model(model, loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {acc * 100:.2f}%")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusión")
    plt.xlabel("Etiqueta Predicha")
    plt.ylabel("Etiqueta Verdadera")
    plt.tight_layout()
    plt.show()

train_model(model, train_loader, epochs=15)
class_names = [class_descriptions[index_to_code[i]] for i in range(len(index_to_code))]
evaluate_model(model, test_loader, class_names)