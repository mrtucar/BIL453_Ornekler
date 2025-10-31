# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 21:56:57 2025

@author: murat.ucar
"""

#%% L1 ORnek
import torch
import torch.nn as nn
import torch.optim as optim

# Modeli ve optimizer'ı tanımla (bu kez weight_decay OLMADAN)
model = nn.Linear(64, 32)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss() # Örnek bir kayıp fonksiyonu

# L1 ceza katsayısı
l1_lambda = 0.005

# --- Örnek bir eğitim adımı ---
# inputs, targets = ... (verinizin geldiğini varsayalım)
#
# optimizer.zero_grad()
#
# outputs = model(inputs)
# loss = criterion(outputs, targets) # Ana kaybı hesapla

# L1 cezasını manuel olarak hesapla
l1_norm = 0
for param in model.parameters():
    l1_norm += torch.sum(torch.abs(param))

# L1 cezasını ana kayba ekle
total_loss = loss + l1_lambda * l1_norm

# Geri yayılımı toplam kayıp üzerinden yap
# total_loss.backward()
# optimizer.step()

#%%
#L2 Ornek
import torch.optim as optim
import torch.nn as nn

# Önce modelinizi tanımlayın (örneğin)
model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10)
)

# L2 Düzenlileştirmesini (Weight Decay) optimizer'da belirtin
# weight_decay=0.01 : L2 ceza katsayısını belirler.
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Eğitim döngünüzde başka bir değişiklik yapmanıza gerek yoktur.
# loss.backward() çağrıldığında optimizer gradyanlara L2 cezasını
# otomatik olarak uygular.