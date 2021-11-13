#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/gibranfp/CursoAprendizajeProfundo/blob/2022-1/notebooks/4a_ucf11_rnn_class.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Reconocimiento de acciones humanas usando RNNs 
# 
# Curso: [Aprendizaje Profundo](http://turing.iimas.unam.mx/~gibranfp/cursos/aprendizaje_profundo/). Profesor: [Gibran Fuentes Pineda](http://turing.iimas.unam.mx/~gibranfp/). Ayudantes: [Bere](https://turing.iimas.unam.mx/~bereml/) y [Ricardo](https://turing.iimas.unam.mx/~ricardoml/).
# 
# 
# ---
# ---
# 
# En esta libreta entrenaremos un modelo basado en RNNs para reconocimiento de acciones humanas (HAR) en el conjunto [UCF11](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php).
# 
# <img src="https://www.cs.ucf.edu/~liujg/realistic_actions/youtube_snaps.jpg" />
# 
# Este ejemplo está basado en las ideas presentadas en [*Long-term Recurrent Convolutional Networks for Visual Recognition and Description*](https://arxiv.org/abs/1411.4389) de 2016 por Donahue et al. 

# ## 1 Preparación

# ### 1.1 Bibliotecas

# In[1]:


# Colab
# https://github.com/TylerYep/torchinfo
get_ipython().system('pip install torchinfo')
# https://zarr.readthedocs.io/en/stable/
get_ipython().system('pip install zarr')


# In[2]:


# sistema de archivos
import os
# funciones aleatorias
import random
# descomprimir
import tarfile
# sistema de archivos
from os.path import join

# arreglos multidimensionales
import numpy as np
# redes neuronales
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets.utils as tvu
# almacenamiento de arreglos multidimensionales
import zarr
#redes
from torch.utils.data import DataLoader, random_split
# inspección de arquitectura
from torchinfo import summary

# directorio de datos
DATA_DIR = '../data'

# tamaño del lote
BATCH_SIZE = 32
# tamaño del vector de características
FEAT_SIZE = 1024

# reproducibilidad
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch_gen = torch.manual_seed(SEED)


# ## 2 Datos

# ### 2.1 Conjunto de datos

# In[3]:


class UCF11:

    def __init__(self, root, download=False):
        self.root = root
        self.zarr_dir = join(root, 'ucf11.zarr')
        if download:
            self.download()
        self.z = zarr.open(self.zarr_dir, 'r')
        self.paths = list(self.z.array_keys())
        
    def __getitem__(self, i):
        arr = self.z[self.paths[i]]
        x = np.array(arr)
        y = np.array(arr.attrs['y'], dtype=np.int64)
        return x, y

    def __len__(self):
        return len(self.paths)
    
    def _check_integrity(self):
        return os.path.isdir(self.zarr_dir)
    
    def _extract(self, root, filename):
        tar = tarfile.open(join(root, filename), "r:gz")
        tar.extractall(root)
        tar.close()

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        tvu.download_url(
            url='https://cloud.xibalba.com.mx/s/apYrNA4iM4K65o7/download',
            root=self.root,
            filename='ucf11.zarr.tar.gz',
            md5='c8a82454f9ec092d00bcd99c849e03fd'
        )
        self._extract(self.root, 'ucf11.zarr.tar.gz')


# ### 2.2 Instancia del conjunto y partición

# In[4]:


ds = UCF11(join(DATA_DIR, 'ucf11'), True)
x, y = ds[0]
print(f'x shape={x.shape} dtype={x.dtype}')
print(f'x [0][:5]={x[0][:5]}')
print(f'y shape={y.shape} dtype={y.dtype} {y}')
print(f'y {y}')


# In[5]:


trn_size = int(0.8 * len(ds))
tst_size = len(ds) - trn_size
trn_ds, tst_ds = random_split(ds, [trn_size, tst_size])
len(trn_ds), len(tst_ds)


# ### 2.3 Cargadores de datos

# In[6]:


trn_dl = DataLoader(
    # conjunto
    trn_ds,
    # tamaño del lote
    batch_size=BATCH_SIZE,
    # desordenar
    shuffle=True,
    # procesos paralelos
    num_workers=2
)
tst_dl = DataLoader(
    # conjunto
    tst_ds,
    # tamaño del lote
    batch_size=BATCH_SIZE,
    # desordenar
    shuffle=True,
    # procesos paralelos
    num_workers=2
)


# In[7]:


x, y = next(iter(trn_dl))
print(f'x shape={x.shape} dtype={x.dtype}')
print(f'y shape={y.shape} dtype={y.dtype}')


# ## 3 Modelo
# 
# <!-- Torchvision provee una familia de [modelos](https://pytorch.org/docs/1.6.0/torchvision/models.html#classification) preentrenados en ImageNet. Usaremos [Shufflenet V2](https://arxiv.org/abs/1807.11164), una arquitectura eficiente para clasificación de imágenes.  -->

# ### 3.1 Definición de arquitectura

# In[8]:


class RNN(nn.Module):

    def __init__(self, input_size=1024, hidden_size=128, num_classes=11):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=True)
        self.cls = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Batch, Seq, Feats, Hidden
        # [B, S, F] => [B, F, S]
        x = x.movedim(1, 2)
        # [B, F, S]
        x = self.bn(x)
        # [B, F, S] => [B, S, F]
        x = x.movedim(1, 2)
        # [B, S, F] => [B, S, H]
        x, _ = self.rnn(x)
        # [B, S, H] => [B, H]
        # toma el último paso, participación 1
        x = x[:, -1, :]
        # [B, H] = [B, 11]
        x = self.cls(x)
        return x


# In[9]:


model = RNN().eval()
model(torch.zeros(1, 10, 1024)).shape


# ### 3.2 Inspección de arquitectura

# In[10]:


summary(model, (1, 10, 1024), device='cpu', verbose=0)


# ## 4 Entrenamiento

# ### 4.1 Ciclo de entrenamiento

# In[11]:


# optimizador
opt = optim.Adam(model.parameters(), lr=1e-3)

# ciclo de entrenamiento
EPOCHS = 10
for epoch in range(EPOCHS):

    # modelo en modo de entrenamiento
    model.train()
    
    # entrenamiento de una época
    for x, y_true in trn_dl:
        # hacemos inferencia para obtener los logits
        y_lgts = model(x)
        # calculamos la pérdida
        loss = F.cross_entropy(y_lgts, y_true)
        # vaciamos los gradientes
        opt.zero_grad()
        # retropropagamos
        loss.backward()
        # actulizamos parámetros
        opt.step()

    # desactivamos temporalmente la gráfica de cómputo
    with torch.no_grad():

        # modelo en modo de evaluación
        model.eval()
        
        losses, accs = [], []
        # validación de la época
        for x, y_true in tst_dl:
            # hacemos inferencia para obtener los logits
            y_lgts = model(x)
            # calculamos las probabilidades
            y_prob = F.softmax(y_lgts, 1)
            # obtenemos la clase predicha
            y_pred = torch.argmax(y_prob, 1)
            
            # calculamos la pérdida
            loss = F.cross_entropy(y_lgts, y_true)
            # calculamos la exactitud
            acc = (y_true == y_pred).type(torch.float32).mean()

            # guardamos históricos
            losses.append(loss.item() * 100)
            accs.append(acc.item() * 100)

        # imprimimos métricas
        loss = np.mean(losses)
        acc = np.mean(accs)
        print(f'E{epoch:2} loss={loss:6.2f} acc={acc:.2f}')


# ## Participación 1
# 
# Cambia la arquitectura para tomar el promedio de la secuencia de la última salida de la capa RNN en vez de tomar el último paso. Revisa la documentación de [`torch.mean`](https://pytorch.org/docs/stable/generated/torch.mean.html)

# ## Participación 2
# 
# Remplaza la capa [GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) por una [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html).
