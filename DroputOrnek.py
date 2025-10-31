# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 21:48:23 2025

@author: murat.ucar
"""



#%%
import torch.nn as nn

# Model yapısını tanımlıyoruz
# Dropout genellikle tam bağlı (Linear) katmanlar arasına yerleştirilir.

model_with_dropout = nn.Sequential(
    nn.Linear(784, 128),  # Giriş katmanı
    nn.ReLU(),
    nn.Dropout(p=0.5),    # Nöronların %50'sini rastgele sıfırlar (sadece eğitim sırasında)
    nn.Linear(128, 10),    # Çıkış katmanı
    nn.Softmax(dim=1)
)

# Modeli yazdır
print(model_with_dropout)