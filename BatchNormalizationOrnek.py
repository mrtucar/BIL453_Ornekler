# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 22:09:31 2025

@author: murat.ucar
"""
#%% 2D CNN Örneği
import torch.nn as nn

# Girdi verisi [Batch_Size, Kanal_Sayısı, Yükseklik, Genişlik] şeklindedir
# Örn: [32, 3, 28, 28]

# 3 giriş kanalı (RGB), 16 çıkış kanalı olan bir Conv katmanı
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

# BatchNorm2d, 'kanal sayısını' (yani 16'yı) argüman olarak alır
batch_norm_layer = nn.BatchNorm2d(16)

activation_layer = nn.ReLU()

# Modelde kullanım sırası: Conv -> BatchNorm -> ReLU
model_cnn = nn.Sequential(
    conv_layer,
    batch_norm_layer,
    activation_layer
)

print("--- CNN (2D) Örneği ---")
print(model_cnn)
#%%
import torch.nn as nn

# Girdi verisi [Batch_Size, Özellik_Sayısı] şeklindedir
# Örn: [32, 128]

# 128 giriş özelliği, 64 çıkış özelliği olan bir Linear katman
linear_layer = nn.Linear(in_features=128, out_features=64)

# BatchNorm1d, 'özellik sayısını' (yani 64'ü) argüman olarak alır
batch_norm_layer_1d = nn.BatchNorm1d(64)

activation_layer = nn.ReLU()

# Modelde kullanım sırası: Linear -> BatchNorm -> ReLU
model_fc = nn.Sequential(
    linear_layer,
    batch_norm_layer_1d,
    activation_layer
)

print("\n--- Tam Bağlı (1D) Örneği ---")
print(model_fc)