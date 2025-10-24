# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 15:26:42 2025

@author: murat.ucar
"""
#%%
import torchvision.models as models
#pretrained özelliği modelin imagenet 
#verileri ile eğitim sonucundaki ağırlıklarını yükler
alexnet = models.alexnet(pretrained=True)
#%% 
#Bir görüntü için tahmin üretme
import torch
import torch
from torchvision import models, transforms
from PIL import Image

#Normalize => output = (input - mean) / std
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                        )
])

model = models.alexnet(pretrained=True)
model.eval()  # Modeli çıkarım (inference) moduna ayarla

img = Image.open('OrnekResimler/kedi.jpg')
img_t = preprocess(img)
batch_t = img_t.unsqueeze(0)

with torch.no_grad():  # Gradyan hesaplamasını devre dışı bırakır.
    output = model(batch_t)

probs = torch.nn.functional.softmax(output, dim=1)
top_p, top_class = probs.topk(1, dim=1)

print(f'Tahmin Edilen Sınıf İndeksi: {top_class.item()}')
print(f'Olasılık: {top_p.item():.4f}')
#%%
with open("imageNetClasses.txt") as f:
    idx2label = eval(f.read())

print(idx2label[top_class.item()])


