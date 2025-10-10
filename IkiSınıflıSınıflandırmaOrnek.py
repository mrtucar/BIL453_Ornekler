# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:03:56 2025

@author: murat.ucar
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 09:48:54 2025

@author: murat.ucar
"""

#%%
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

#%%
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 1. Veri Setini Yükle
breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

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
class BreastCancerDataset(Dataset):
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
train_dataset = BreastCancerDataset(X_train_tensor, y_train_tensor)
test_dataset = BreastCancerDataset(X_test_tensor, y_test_tensor)



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
        # (Binary sınıflandırma için sigmoid aktivasyon fonksiyonu)
        out =torch.sigmoid( self.fc3(x)) 
        return out

# Modelin Örneğini Oluşturma
input_size = 30       
num_classes = 1    
model = SimpleClassifier(input_size, num_classes)

#%%
sonuc = model(train_dataset.__getitem__(10)[0])
print(sonuc)
print(train_dataset.__getitem__(10)[1])

#%%
# Hiperparametreler
learning_rate = 0.01
num_epochs = 100

# Kayıp Fonksiyonu: İki sınıflı sınıflandırma için Binary Cross Entropi (BCELoss )
criterion = nn.BCELoss()
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
model.eval() # Değerlendirme moduna ayarla
with torch.no_grad():
    total = 0
    correct = 0
    
    for inputs, labels in test_loader:
        outputs = model(inputs)
        
        # Tahminleri oluştur: Olasılık >= 0.5 ise 1, aksi takdirde 0
        # outputs [N, 1] olduğu için squeeze() kullanıldı.
        predicted = (outputs.data > 0.5).float().squeeze() 
        
        # labels de [N, 1] olduğu için kıyaslama yapmadan önce sıkıştır
        labels = labels.squeeze()

        total += labels.size(0)
        # Tahminler ile gerçek etiketleri karşılaştır
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nModelin Eğitim Veri Seti Doğruluğu: {accuracy:.2f}%')

#%%
random_inputs = torch.rand(30) 
tahmin = model(random_inputs)
print("Tahmin Edilen Değer", tahmin)

