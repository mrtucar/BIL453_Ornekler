# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 09:48:54 2025

@author: murat.ucar
"""

#%%
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

#%%
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 1. Veri Setini Yükle
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# 2. Veri Setini Eğitim ve Test Kümelerine Ayır (Standart Uygulama)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Verileri PyTorch Tensörlerine Dönüştür
# Özellikler (X) için float32, Etiketler (y) için long (tam sayı) ku<llanırız.
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
#%%
# Özel Dataset Sınıfı
class DiabetDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __len__(self):
        # Veri setindeki toplam örnek sayısı
        return len(self.X_data)

    def __getitem__(self, index):
        # Belirtilen indeksteki örneği ve etiketini döndürür
        return self.X_data[index], self.y_data[index]

# Dataset Örneklerini Oluşturma
train_dataset = DiabetDataset(X_train_tensor, y_train_tensor)
test_dataset = DiabetDataset(X_test_tensor, y_test_tensor)



#%%
# DataLoader Oluşturma
# DataLoader, verileri mini-partiler (mini-batches) halinde sunarak eğitimi hızlandırır.
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16)
#%%
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        # 1. Katman: 4 giriş -> 10 gizli nöron
        self.fc1 = nn.Linear(input_size, 10)
        # 2. Katman: 10 gizli nöron -> 10 gizli nöron
        self.fc2 = nn.Linear(10, 10)
        # 3. Katman (Çıkış): 10 gizli nöron -> 3 çıkış nöronu (sınıf sayısı)
        self.fc3 = nn.Linear(10, num_classes)

    def forward(self, x):
        # ReLU aktivasyon fonksiyonunu uygula
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # Çıktı katmanı (sınıflandırma için aktivasyon yok, çünkü Kayıp Fonksiyonu (Loss) bunu halledecek)
        out = self.fc3(x) 
        return out

# Modelin Örneğini Oluşturma
input_size = 10        # sepal uzunluğu, genişliği, petal uzunluğu, genişliği
num_classes = 1     # 3 farklı iris türü
model = SimpleClassifier(input_size, num_classes)

#%%
sonuc = model(train_dataset.__getitem__(10)[0])
print(sonuc)
print(train_dataset.__getitem__(10)[1])

#%%
# Hiperparametreler
learning_rate = 0.01
num_epochs = 100

# Kayıp Fonksiyonu: Mean Squared Error
criterion = nn.MSELoss()
# Optimizasyon: Adam algoritması
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("\n--- Model Eğitiliyor (Training Starts) ---")

for epoch in range(num_epochs):
    for features, labels in train_loader:
        # 1. İleri Yayılım (Forward pass)
        outputs = model(features)
        loss = criterion(outputs, labels.reshape(-1,1))

        # 2. Geri Yayılım ve Optimizasyon (Backward and optimize)
        optimizer.zero_grad() # Önceki gradyanları sıfırla
        loss.backward()       # Gradyanları hesapla
        optimizer.step()      # Model parametrelerini güncelle

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n--- Eğitim Tamamlandı (Training Finished) ---")
#%%
import torch.nn.functional as F # Kayıp fonksiyonları için

# Modeli değerlendirme moduna ayarla
model.eval()

total_mse = 0.0
total_samples = 0

with torch.no_grad(): # Gradyan hesaplamalarını devre dışı bırak
    for features, labels in test_loader:
        outputs = model(features)

        # 1. MSE'yi hesapla
        # F.mse_loss(tahminler, gerçek_değerler, reduction='sum') toplam hatayı verir
        mse_loss = F.mse_loss(outputs, labels.reshape(-1,1), reduction='sum')
        
        total_mse += mse_loss.item()
        # Etiket sayısını toplama ekle
        total_samples += labels.size(0)

# Ortalama Karesel Hata (MSE)
mean_squared_error = total_mse / total_samples

# Kök Ortalama Karesel Hata (RMSE) (yaygın olarak raporlanır)
# RMSE, MSE'nin kareköküdür.
rmse = torch.sqrt(torch.tensor(mean_squared_error)).item()

print(f'\nModelin Test Veri Seti MSE (Ortalama Karesel Hata): {mean_squared_error:.4f}')
print(f'Modelin Test Veri Seti RMSE (Kök Ortalama Karesel Hata): {rmse:.4f}')

#%%
random_inputs = torch.rand(10) 
tahmin = model(random_inputs)
print("Tahmin Edilen Değer", tahmin)

