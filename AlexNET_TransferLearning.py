# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:14:38 2025

@author: murat.ucar
"""

#%%
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# CIFAR-10 için özel transform (224x224'e büyütme dahil)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # ImageNet normalizasyon değerleri kullanılır
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Veri setini yükle ve veri yükleyicisini (DataLoader) oluştur
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(trainset, batch_size=64, shuffle=False)

#%%
# AlexNet'i ImageNet ağırlıklarıyla yükle
model = models.alexnet(pretrained=True,)

# Önemli Adım: Ağırlıkları Dondur (Sadece son katmanı eğiteceğiz - Özellik Çıkarma)
for param in model.parameters():
    param.requires_grad = False

# Son tam bağlantılı katmanı (classifier) 10 sınıfa uyarla
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 10) # Çıktı 10 sınıf olacak
#%%
# Sadece 'model.classifier[6]' parametrelerini optimize et (Dondurulmamış tek katman)
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Eğitim başlıyor. Cihaz: {device}")

for epoch in range(5):  # 5 epochluk kısa bir eğitim
    running_loss = 0.0
    
    # Her mini-batch için döngü
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Gradyanları sıfırla
        optimizer.zero_grad()
        
        # İleri yayılım (Forward Pass)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Geriye yayılım (Back Propagation) ve Ağırlık Güncelleme
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}, Kayıp (Loss): {running_loss/len(trainloader):.4f}")

print("Eğitim tamamlandı.")
#%%
# Modeli değerlendirme moduna ayarla
model.eval()

with torch.no_grad(): # Gradyan hesaplamalarını devre dışı bırak (bellek tasarrufu ve hız için)
    correct = 0
    total = 0
    for features, labels in testloader:
        features = features.to(device)
        labels = labels.to(device)
        
        outputs = model(features)
        # En yüksek olasılığa sahip sınıfı tahmin et
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nModelin Test Veri Seti Doğruluğu: {accuracy:.2f}%')
    