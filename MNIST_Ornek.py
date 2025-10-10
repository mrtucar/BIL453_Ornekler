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

transform = transforms.Compose([
    # Görüntüyü (PIL Image) [0, 255] aralığından [0.0, 1.0] aralığında
    # bir PyTorch Tensor'una (şekli: [C, H, W]) dönüştürür.
    transforms.ToTensor()
])

train_set = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=1,shuffle=True)

for images, labels in train_loader:
    
    # Bu döngüdeki ilk iterasyonda (yani ilk batch'te) dur
    single_image_data = images
    single_label_data = labels
    
    break

print(f"Alınan Görüntü Şekli: {single_image_data.shape}")
print(f"Alınan Etiket Değeri: {single_label_data}")

plt.figure(figsize=(4, 4))
# Gri tonlamalı görüntü için cmaps'iz Tensor gösterilir.
plt.imshow(single_image_data.numpy().squeeze(), cmap='gray') 
plt.title(f"Etiket: {single_label_data}", fontsize=15)
plt.axis('off')
plt.show()

#images,labels = next(iter(train_loader))
#%% 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor()
])

train_set = torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform)
test_set = torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform)

train_loader = torch.utils.data.DataLoader(train_set,batch_size=32,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=32)

#%%
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        #Görüntüler iki boyutlu olarak geliyor öncelikle bunu tek boyuta indiriyoruz.
        self.flatten = nn.Flatten()
        # 1. Katman: 28*28 giriş -> 784 Giriş Nöronu
        self.fc1 = nn.Linear(input_size, 128)
        # 2. Katman: 10 gizli nöron -> 10 gizli nöron
        self.fc2 = nn.Linear(128, 64)
        # 3. Katman (Çıkış): 10 gizli nöron -> 3 çıkış nöronu (sınıf sayısı)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # ReLU aktivasyon fonksiyonunu uygula
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Çıktı katmanı (sınıflandırma için aktivasyon yok, çünkü Kayıp Fonksiyonu (Loss) bunu halledecek)
        out = self.fc3(x) 
        return out

# Modelin Örneğini Oluşturma
input_size = 28*28        
num_classes = 10    
model = SimpleClassifier(input_size, num_classes)
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
        outputs = model(features)
        # En yüksek olasılığa sahip sınıfı tahmin et
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nModelin Test Veri Seti Doğruluğu: {accuracy:.2f}%')