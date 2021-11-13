
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
# redes
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
# inspección de arquitectura
from torchinfo import summary
from itertools import islice as take

from tqdm import tqdm


def train_epoch(dl, model, opt, device):
    # modelo en modo de entrenamiento
    model.train()

    # entrenamiento de una época
    for x, y_true in dl:
        x = x.to(device)
        y_true = y_true.to(device)
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


def eval_epoch(dl, model, device, num_batches=None):
    # desactivamos temporalmente la gráfica de cómputo
    with torch.no_grad():
        # modelo en modo de evaluación
        model.eval()

        losses, accs = [], []
        # validación de la época
        for x, y_true in dl:
            x = x.to(device)
            y_true = y_true.to(device)
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

        return loss, acc


# ### 4.1 Ciclo de entrenamiento

def train_model(trn_dl, tst_dl, model, epochs=10):
    # optimizador
    opt = optim.Adam(model.parameters(), lr=1e-3)

    # usamos GPU si está disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # movemos a dispositivo
    model.to(device)

    loss_hist, acc_hist = [], []
    # ciclo de entrenamiento
    for epoch in tqdm(range(epochs)):
        train_epoch(trn_dl, model, opt, device)

        # evaluamos la época en entrenamiento
        trn_loss, trn_acc = eval_epoch(trn_dl, model, device)

        # evaluamos la época en validación
        val_loss, val_acc = eval_epoch(tst_dl, model, device)

        loss_hist.append([trn_loss, val_loss])
        acc_hist.append([trn_acc, val_acc])

    loss_hist = np.array(loss_hist)
    acc_hist = np.array(acc_hist)

    return {'trn_loss_hist': loss_hist[:, 0],
            'tst_loss_hist': loss_hist[:, 1],
            'trn_acc_hist': acc_hist[:, 1],
            'tst_acc_hist': acc_hist[:, 0]}

def history_plot(history, features=['trn_loss_hist', 'tst_loss_hist']):
    fig, ax = plt.subplots(figsize=(8, 6))
    for feature in features:
        ax.plot(history[feature], label=feature)

    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.legend()
    plt.title('Loss History')
    plt.show()

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


if __name__ == '__main__':
    # %%
    # directorio de datos
    DATA_DIR = os.path.join('..', 'data')

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

    # ### 2.2 Instancia del conjunto y partición

    ds = UCF11(join(DATA_DIR, 'ucf11'), True)
    x, y = ds[0]
    print(f'x shape={x.shape} dtype={x.dtype}')
    print(f'x [0][:5]={x[0][:5]}')
    print(f'y shape={y.shape} dtype={y.dtype} {y}')
    print(f'y {y}')

    trn_size = int(0.8 * len(ds))
    tst_size = len(ds) - trn_size
    trn_ds, tst_ds = random_split(ds, [trn_size, tst_size])
    print(len(trn_ds), len(tst_ds))

    # %% ### 2.3 Cargadores de datos

    trn_dl = DataLoader(
        # conjunto
        trn_ds,
        # tamaño del lote
        batch_size=BATCH_SIZE,
        # desordenar
        shuffle=True,
        # procesos paralelos
        num_workers=0
    )
    tst_dl = DataLoader(
        # conjunto
        tst_ds,
        # tamaño del lote
        batch_size=BATCH_SIZE,
        # desordenar
        shuffle=True,
        # procesos paralelos
        num_workers=0
    )

    x, y = next(iter(trn_dl))
    print(f'x shape={x.shape} dtype={x.dtype}')
    print(f'y shape={y.shape} dtype={y.dtype}')


    # ## 3 Modelo
    #
    # <!-- Torchvision provee una familia de [modelos](https://pytorch.org/docs/1.6.0/torchvision/models.html#classification) preentrenados en ImageNet. Usaremos [Shufflenet V2](https://arxiv.org/abs/1807.11164), una arquitectura eficiente para clasificación de imágenes.  -->

    # ### 3.1 Definición de arquitectura

    # In[8]:

    class RNN(nn.Module):

        def __init__(self, input_size=1024, hidden_size=128, num_classes=11, bidirectional=True):
            super().__init__()
            self.bn = nn.BatchNorm1d(input_size)
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=1, batch_first=True, bidirectional=True)
            self.cls = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)

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

    model = RNN().eval()
    out = model(torch.zeros(1, 10, 1024))
    print(out.shape)

    # %% ### 3.2 Inspección de arquitectura
    print(summary(model, (1, 10, 1024), device='cpu', verbose=0))

    # %% ## 4 Entrenamiento
    model = RNN(bidirectional=True)
    history = train_model(trn_dl, tst_dl, model, epochs=30)
    history_plot(history)

    #%%
    class CNN(nn.Module):

        def __init__(self, input_size=1024, channels=10, kernel=3, hidden_size=128, num_classes=11):
            super().__init__()

            self.bn = nn.BatchNorm1d(input_size)
            self.cnn = nn.Conv1d(in_channels=input_size,
                                 out_channels=hidden_size,
                                 kernel_size=kernel,
                                 padding='same')

            self.dropout = nn.Dropout(p=0.3)

            self.cls = nn.Linear(hidden_size * channels, num_classes)

        def forward(self, x):
            # Batch, Seq, Feats, Hidden
            # [B, S, F] => [B, F, S]
            x = x.movedim(1, 2)
            # [B, F, S]
            x = self.bn(x)
            # [B, F, S] => [B, S, F]
            # x = x.movedim(1, 2)
            # [B, S, F] => [B, S, H]
            x = self.cnn(x)

            x = self.dropout(x)
            # [B, S, H] => [B, H]
            x = torch.flatten(x, start_dim=1, end_dim=2)
            x = self.cls(x)
            return x

    model = CNN(hidden_size=250).eval()
    out = model(torch.zeros(1, 10, 1024))
    print(out.shape)
    print(summary(model, (1, 10, 1024), device='cpu', verbose=0))

    # %% ## 4 Entrenamiento
    model = CNN(hidden_size=80)
    history = train_model(trn_dl, tst_dl, model, epochs=30)
    #%%
    history_plot(history, features=['trn_loss_hist', 'tst_loss_hist'])
    history_plot(history, features=['trn_acc_hist', 'tst_acc_hist'])

    # #%%
    # m = nn.Conv1d(10, 128, 3, padding='same')
    # input = torch.randn(1, 10, 1024)
    # output = m(input)
    # print(output.shape)