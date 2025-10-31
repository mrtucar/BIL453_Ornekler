# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 21:59:59 2025

@author: murat.ucar
"""

#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np # best_val_loss'u başlatmak için

# --- 1. Erken Durdurma Parametrelerini Ayarlama ---

# Kaç epoch boyunca iyileşme olmazsa durdurulacak?
patience = 10 

# Sabır sayacı
patience_counter = 0

# En iyi doğrulama kaybını saklamak için (sonsuz ile başlatılır)
best_val_loss = np.Inf

# Modelinizin, optimizer'ınızın ve kayıp fonksiyonunuzun olduğunu varsayalım
# model = ...
# optimizer = ...
# criterion = ...
# train_loader = ...
# val_loader = ...

num_epochs = 100 # Maksimum epoch sayısı

print("Eğitim başlıyor...")

# --- 2. Eğitim Döngüsü ---
for epoch in range(num_epochs):
    
    # --- Eğitim Aşaması ---
    model.train() # Modeli eğitim moduna al (Dropout vb. aktif)
    running_train_loss = 0.0
    
    for data in train_loader:
        # inputs, labels = data
        # optimizer.zero_grad()
        # outputs = model(inputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
        # running_train_loss += loss.item()
        pass # (Yukarıdaki kodun çalıştığını varsayalım)

    avg_train_loss = running_train_loss / len(train_loader)

    # --- Doğrulama Aşaması ---
    model.eval() # Modeli değerlendirme moduna al (Dropout vb. pasif)
    running_val_loss = 0.0
    
    with torch.no_grad(): # Gradyan hesaplamayı kapat
        for data in val_loader:
            # inputs, labels = data
            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # running_val_loss += loss.item()
            pass # (Yukarıdaki kodun çalıştığını varsayalım)

    avg_val_loss = running_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Eğitim Kaybı: {avg_train_loss:.4f} | "
          f"Doğrulama Kaybı: {avg_val_loss:.4f}")

    # --- 3. Erken Durdurma Mantığı ---
    
    if avg_val_loss < best_val_loss:
        # İyileşme var, en iyi kaybı güncelle ve modeli kaydet
        best_val_loss = avg_val_loss
        patience_counter = 0 # Sayacı sıfırla
        
        # En iyi modelin ağırlıklarını kaydet
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"   -> İyileşme var, model 'best_model.pth' olarak kaydedildi.")
        
    else:
        # İyileşme yok, sabır sayacını artır
        patience_counter += 1
        print(f"   -> İyileşme yok. Sabır: {patience_counter}/{patience}")

    # Eğer sabır sayacı 'patience' limitine ulaştıysa, eğitimi durdur
    if patience_counter >= patience:
        print(f"\n--- ERKEN DURDURMA ---")
        print(f"Doğrulama kaybı {patience} epoch boyunca iyileşmedi. Eğitim durduruluyor.")
        break # Döngüyü kır

# --- 4. Eğitim Sonrası ---
print("\nEğitim tamamlandı.")

# En iyi ağırlıkları yükleyerek modeli en iyi performansına geri döndür
print(f"En iyi model ağırlıkları 'best_model.pth' dosyasından yükleniyor...")
# model.load_state_dict(torch.load('best_model.pth'))

# Artık 'model' en iyi doğrulama performansına sahip haliyle test seti üzerinde kullanılabilir.