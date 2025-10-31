# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 21:33:05 2025

@author: murat.ucar
"""

#%%
# Gerekli kütüphaneleri import edelim
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Görselleştirme için daha fazla artırma içeren bir pipeline
visualization_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), # Daha agresif kırpma
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30), # 30 dereceye kadar rastgele döndürme
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Renklerle oynama
    # transforms.ToTensor() -> PIL Image olarak kalması plt.imshow için daha kolay
])

# Örnek olarak bir kedi görüntüsü yükleyelim
# (Bu satırı çalıştırmak için 'kedi.jpg' adında bir dosyanız olmalı)
# Veya internetten bir görüntü indirin
# https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg

img = Image.open('Hafta_03/OrnekResimler/kedi.jpg')

plt.figure(figsize=(15, 8))

# Orijinal görüntü
plt.subplot(2, 4, 1)
plt.imshow(img)
plt.title("Orijinal Görüntü")
plt.axis('off')

# Aynı görüntüye 7 farklı artırma uygula
for i in range(7):
    augmented_img = visualization_transform(img)
    plt.subplot(2, 4, i + 2)
    plt.imshow(augmented_img)
    plt.title(f"Artırma #{i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()