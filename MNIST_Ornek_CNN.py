# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:37:13 2025

@author: murat.ucar
"""

#%% 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# CUDA varsa 'cuda' (GPU), yoksa 'cpu' kullan.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")

transform = transforms.Compose([
    transforms.ToTensor()
])

train_set = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform)
test_set = torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=32,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=32)

#%%
class SimpleClassifier(nn.Module):
    def __init__(self,  num_classes):
        super(SimpleClassifier, self).__init__()
        # 1. Evrişim ve Havuzlama Bloğu
        # Giriş: [Batch, 1, 28, 28]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        # Çıkış: [Batch, 16, 28, 28] (Padding sayesinde aynı kalır)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Çıkış: [Batch, 16, 14, 14]

        # 2. Evrişim ve Havuzlama Bloğu
        # Giriş: [Batch, 16, 14, 14]
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        # Çıkış: [Batch, 32, 14, 14]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Çıkış: [Batch, 32, 7, 7]

        # Tam Bağlantılı (Fully Connected) Katmanlar
        # Düzleştirme sonrası giriş boyutu: 32 kanalı * 7 yükseklik * 7 genişlik = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128) 
        self.fc2 = nn.Linear(128, num_classes)
        
        # Aktivasyon fonksiyonu (ReLU) ve Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Evrişim Katmanları
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        
        # Düzleştirme (Flattening): FC katmanlara geçiş
        # x.size(0) = Batch boyutu, -1 kalan tüm boyutları düzleştirir (32*7*7)
        x = x.view(x.size(0), -1) 
        
        # Tam Bağlantılı Katmanlar
        x = self.dropout(self.relu(self.fc1(x)))
        out = self.fc2(x) 
        return out

# Modelin Örneğini Oluşturma
num_classes = 10    
model = SimpleClassifier(num_classes)

model.to(device)

#%%
# Hiperparametreler
learning_rate = 0.01
num_epochs = 10

# Kayıp Fonksiyonu: Çok sınıflı sınıflandırma için Çapraz Entropi (Cross-Entropy Loss)
criterion = nn.CrossEntropyLoss()
# Optimizasyon: Adam algoritması
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("\n--- Model Eğitiliyor (Training Starts) ---")

for epoch in range(num_epochs):
    for features, labels in train_loader:
        
        features = features.to(device)
        labels = labels.to(device)
        
        # 1. İleri Yayılım (Forward pass)
        outputs = model(features)
        loss = criterion(outputs, labels)

        # 2. Geri Yayılım ve Optimizasyon (Backward and optimize)
        optimizer.zero_grad() # Önceki gradyanları sıfırla
        loss.backward()       # Gradyanları hesapla
        optimizer.step()      # Model parametrelerini güncelle

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n--- Eğitim Tamamlandı (Training Finished) ---")
#%%
# Modeli değerlendirme moduna ayarla
model.eval()

with torch.no_grad(): # Gradyan hesaplamalarını devre dışı bırak (bellek tasarrufu ve hız için)
    correct = 0
    total = 0
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        outputs = model(features)
        # En yüksek olasılığa sahip sınıfı tahmin et
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nModelin Test Veri Seti Doğruluğu: {accuracy:.2f}%')