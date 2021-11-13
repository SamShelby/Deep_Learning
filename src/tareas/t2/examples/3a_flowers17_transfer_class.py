#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/gibranfp/CursoAprendizajeProfundo/blob/2022-1/notebooks/3a_flowers17_transfer_class.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Clasificación de grano fino empleando transferencia de conocimiento
# 
# Curso: [Aprendizaje Profundo](http://turing.iimas.unam.mx/~gibranfp/cursos/aprendizaje_profundo/). Profesor: [Gibran Fuentes Pineda](http://turing.iimas.unam.mx/~gibranfp/). Ayudantes: [Bere](https://turing.iimas.unam.mx/~bereml/) y [Ricardo](https://turing.iimas.unam.mx/~ricardoml/).
# 
# ---
# ---
# 
# En esta libreta veremos un ejemplo sencillo pero completo para clasificación de flores sobre el conjunto [17 Flowers](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/) usando transferencia de conocimiento. El conjunto está formado por 1360 imágenes donde cada una pertenece uno de los 17 tipos de flores. El conjunto provee 3 particiones distintas con 40, 20 y 20 ejemplos de cada flor para entrenamiento, validación y prueba respectivamente. 
# 
# En particular vamos a repasar:
# 
# * conjuntos y cargadores,
# * aumentado de datos/transformaciones,
# * transferencia de conocimiento con [Shufflenet V2](https://arxiv.org/abs/1807.11164),
# * protocolo de entrenamiento y evaluación,
# * monitoreo del entrenamiento,
# * pseudo paro temprano y
# * guardado y carga modelos.
# 
# 
# <img src="https://raw.githubusercontent.com/gibranfp/CursoAprendizajeProfundo/2022-1/figs/flowers17.svg" width="950" height="750" />

# ## 0 Preparación

# In[67]:


# Colab
# get_ipython().system(' pip install torchinfo')


# ### 0.1 Bibliotecas

# In[68]:


# marcas de tiempo
import datetime
# explorar el sistema de archivos
import glob
# sistema de archivos
import os
# funciones aleatorias
import random
# marcas de tiempo
import time
# tomar n elementos de una secuencia
from itertools import islice as take
# sistema de archivos
from os.path import join

# gráficas
import matplotlib.patches as patches
import matplotlib.pyplot as plt
# arreglos multidimensionales
import numpy as np
# csv
import pandas as pd
# redes neuronales
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets.utils as tvu
import torchvision.models as tvm
import torchvision.transforms as T
# leer archivo matlab
from IPython import get_ipython
from scipy.io import loadmat
# cargador de datos
from torch.utils.data import DataLoader
# trazas de Tensorboard
from torch.utils.tensorboard import SummaryWriter
# inspección de arquitectura
from torchinfo import summary
# barra de progreso
# from tqdm.notebook import trange
from tqdm import tqdm
# imágenes
from PIL import Image


# ### 0.2 Auxiliares

# In[69]:

if __name__ == '__main__':

    # directorio de datos
    DATA_DIR = '../data'

    # tamaño del lote
    BATCH_SIZE = 32
    # tamaño de la imagen
    IMG_SIZE = 224

    # filas y columnas de la cuadrícula
    ROWS, COLS = 4, 8


    def display_grid(xs, titles, rows, cols, figsize=(12, 6)):
        """Despliega un ejempos en una cuadrícula."""
        fig, ax = plt.subplots(rows, cols, figsize=figsize)
        i = 0
        for r in range(rows):
            for c in range(cols):
                ax[r, c].imshow(xs[i], cmap='gray')
                ax[r, c].set_title(titles[i])
                ax[r, c].set_xticklabels([])
                ax[r, c].set_yticklabels([])
                i += 1
        fig.tight_layout()
        plt.show()


    def display_batch(x, titles, rows, cols, figsize=(12, 6)):
        """Despliega un lote en una cuadrícula."""
        # denormalizamos
        for c, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
            x[:, c] = x[:, c] * std + mean
        x *= 255
        # rotamos canales
        x = x.permute(0, 2, 3, 1)
        # convertimos a entero
        x = (x.numpy()).astype(np.uint8)
        # desplegamos lote
        display_grid(x, titles, rows, cols, figsize)


    def timestamp(fmt='%y%m%dT%H%M%S'):
        """Regresa la marca de tiempo."""
        return datetime.datetime.fromtimestamp(time.time()).strftime(fmt)


    def set_seed(seed=0):
        """Initializes pseudo-random number generators."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    # reproducibilidad
    set_seed()


    # ## 1. Analizando la arquitectura a transferir
    #
    # Torchvision provee una familia de [modelos](https://pytorch.org/vision/stable/models.html#classification) preentrenados en ImageNet. Los desarrolladores cuidan que los modelos sean entrenados con prácticas estándar y reproducibles. Una buena alternativa es el paquete [`tim`](https://github.com/rwightman/pytorch-image-models) de fast.ai.
    #
    # Usaremos [Shufflenet V2](https://arxiv.org/abs/1807.11164), una arquitectura eficiente para clasificación de imágenes.

    # ### 1.1 Inspeccionando arquitectura

    # Podemos inspeccionar la arquitectura imprimiendo las capas o revisando el código fuente de [ShuffleNet V2](https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py).

    # In[70]:


    model = tvm.shufflenet_v2_x0_5()
    model


    # In[71]:


    summary(model, (1, 3, IMG_SIZE, IMG_SIZE),
            col_names=['input_size', 'output_size'],
            device='cpu', verbose=0)


    # ### 1.2 Remplazando la etapa de clasificación
    #
    # En este caso la etapa de clasificación es solo la capa `model.fc`.

    # In[72]:


    # remplazo de última capa
    model.fc = nn.Linear(1024, 17)

    model


    # In[73]:


    # prueba con datos sintéticos
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(f'{x.shape} => {y.shape}')


    # In[74]:


    summary(model, (1, 3, IMG_SIZE, IMG_SIZE),
            col_names=['input_size', 'output_size'],
            device='cpu', verbose=0)


    # ### 1.3 Estudiando la tubería de datos de la tarea base.
    #
    # Torchvision provee la [información](https://pytorch.org/vision/stable/models.html#classification) necesaria para realizar tranferencia con sus modelos en la página principal (además de los [scripts de entrenamiento](https://github.com/pytorch/vision/tree/main/references/classification) para los curiosos).
    #
    # Para nuestro caso nos interesa el tamaño de la entrada (lo conociamos previamente) y las estádisticas de normalización.

    # In[75]:


    # media y varianza de de ImageNet
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]


    # ## 2 Implementando la tubería de datos para la tarea objetivo.

    # ### 2.1 Tuberias de datos con PyTorch
    #
    # <img src="https://raw.githubusercontent.com/gibranfp/CursoAprendizajeProfundo/2022-1/figs/comida_datos.svg" width="950" height="750" />

    # ### 2.2 Conjunto de datos

    # In[76]:


    class Flowers17:

        def __init__(self, root, split, subset,
                     transform=None, download=False):


            if subset not in {'trn', 'val', 'tst'}:
                ValueError(f'invalid value for subset={subset}')

            # guardamos atributos
            self.root = join(root, 'flowers17')
            self.split = split
            self.subset = subset
            self.transform = transform

            # creamos directorio raíz
            os.makedirs(self.root, exist_ok=True)

            # descargados datos
            if download:
                self.download()

            # verificamos integridad
            if not self._check_integrity():
                raise RuntimeError(
                    'Dataset not found or corrupted.'
                    ' You can use download=True to download it')

            # cargamos partición y subconjunto
            mat = loadmat(join(self.root, 'datasplits.mat'))
            self.x = mat[f'{subset}{split}'][0]

            # leemos las clases y sus números de ejemplos
            df = pd.read_csv(join(self.root, 'classes.csv'))

            # guardamos etiquetas como int
            y = [np.repeat(i, n) for i, n
                 in enumerate(df['examples'])]
            self.y = np.concatenate(y)

            # guardamos etiquetas como str
            self.labels = {i: clazz for i, clazz
                           in enumerate(df['class'])}

        def __getitem__(self, i):
            # cargamos la imagen
            x = self.x[i]
            path = join(self.root, 'jpg', f'image_{x:04d}.jpg')
            img = Image.open(path)
            # aplicamos transformación
            if self.transform is not None:
                img = self.transform(img)
            # leemos la etiqueta como int
            y = self.y[x-1]
            # leemos la etiqueta como str
            label = self.labels[y]
            # regresamos ejemplo como dict
            return {'x': img, 'y': y, 'label': label}

        def __len__(self):
            # regresamos numeros de ejemplos
            return len(self.x)

        def _check_integrity(self):
            return os.path.exists(join(self.root, 'jpg'))

        def download(self):
            if self._check_integrity():
                print('Files already downloaded and verified')
                return
            tvu.download_and_extract_archive(
                url='https://cloud.xibalba.com.mx/s/kG4xHfdGXF3jfiy/download',
                download_root=self.root,
                filename='flowers17.tar.gz',
                md5='8a0e60b25cb39991eda100b66aa57e0a'
            )


    # In[77]:


    ds = Flowers17(DATA_DIR, 1, 'trn', download=True)


    # In[78]:


    len(ds)


    # In[79]:


    example = ds[360]
    x, y, label = example['x'], example['y'], example['label']
    print(f'Imagen dimensiones={x.size} tipo={type(x)}')
    print(f'Etiqueta int={y} str={label}')
    example['x']


    # ### 2.3 Transformaciones
    #
    # Torchvision tiene un conjunto de [transformaciones](https://pytorch.org/docs/1.6.0/torchvision/transforms.html) para ser ejecutadas de forma secuencial cuando se la pasamos a la clase `Compose`.

    # In[80]:


    # transformación de entrenamiento
    trn_tsfm = T.Compose([
        # redimensionamos a Wx224 o 224xH
        T.Resize(IMG_SIZE),
        # cortamos al centro 224x224
        T.CenterCrop(IMG_SIZE),
        # aumentado de datos
        # espejeo horizontal aleatorio
        T.RandomHorizontalFlip(),
        # convertimos a torch.Tensor [3,H,W]
        # escalamos a [0,1]
        T.ToTensor(),
        # estandarizamos con media y varianza
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # conjunto
    trn_ds = Flowers17(DATA_DIR, 1, 'trn', transform=trn_tsfm)


    # In[81]:


    # transformación de validación
    val_tsfm = T.Compose([
        # redimensionamos a Wx224 o 224xH
        T.Resize(IMG_SIZE),
        # cortamos al centro 224x224
        T.CenterCrop(IMG_SIZE),
        # convertimos a torch.Tensor [3,H,W]
        # escalamos a [0,1]
        T.ToTensor(),
        # estandarizamos con media y varianza
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    # conjunto
    val_ds = Flowers17(DATA_DIR, 1, 'val', transform=val_tsfm)


    # ### 2.4 Cargadores de datos

    # Los [cargadores de datos](https://pytorch.org/docs/1.6.0/data.html) pueden precargar el siguiente lote de de entrenamiento forma paralela si especificamos `num_workers => 2`.
    #
    # <img src="https://github.com/gibranfp/CursoAprendizajeProfundo/raw/2022-1/figs/data-loading1.png" />
    # <img src="https://github.com/gibranfp/CursoAprendizajeProfundo/raw/2022-1/figs/data-loading2.png" />
    # Fuente: tensorflow.org

    # In[82]:


    # creamos cargador
    trn_dl = DataLoader(
        # conjunto
        trn_ds,
        # tamaño del lote
        batch_size=BATCH_SIZE,
        # desordenar
        shuffle=True,
        # número de procesos paralelos
        num_workers=0
    )
    # desplegamos lote
    batch = next(iter(trn_dl))
    x, y, labels = batch['x'] , batch['y'], batch['label']
    titles = [f'{v}:{l}' for v, l in zip(y, labels)]
    print(f'x shape={x.shape} dtype={x.dtype}')
    print(f'y shape={y.shape} dtype={y.dtype}')
    display_batch(x, titles, ROWS, COLS)


    # In[83]:


    # creamos cargador
    val_dl = DataLoader(
        # conjunto
        val_ds,
        # tamaño del lote
        batch_size=BATCH_SIZE,
        # desordenar
        shuffle=True,
        # número de procesos paralelos
        num_workers=0
    )
    # desplegamos lote
    batch = next(iter(val_dl))
    x, y, labels = batch['x'] , batch['y'], batch['label']
    titles = [f'{v}:{l}' for v, l in zip(y, labels)]
    print(f'x shape={x.shape} dtype={x.dtype}')
    print(f'y shape={y.shape} dtype={y.dtype}')
    display_batch(x, titles, ROWS, COLS)


    # ## 3 Realizando tranferencia

    # ### 3.1 Carga de pesos

    # In[84]:


    # instancia de modelo y carga de pesos
    model = tvm.shufflenet_v2_x0_5(pretrained=True)


    # ### 3.2 Congelado de parámetros y estadísticas

    # In[85]:


    # congelamos los parámetros
    for param in model.parameters():
        # no participa en la retropropagación
        param.requires_grad = False
    # congelamos las estadísticas
    model.eval()

    # remplazo de última capa
    model.fc = nn.Linear(1024, 17)

    # Nota: hacer esto modifica las estadísticas, cuidado!
    # summary(model, (1, 3, IMG_SIZE, IMG_SIZE),
    #         device='cpu', verbose=0)


    # Empaquetemos la preparación de la arquitectura en una función.

    # In[86]:


    def shufflenet_v2_x0_5(pretrained=True, num_classes=17, freeze_params_buffers=True):
        # instancia e inicilización
        model = tvm.shufflenet_v2_x0_5(pretrained=pretrained)
        if freeze_params_buffers:
            # congelamos los parámetros
            for param in model.parameters():
                # no participa en la retropropagación
                param.requires_grad = False
            # congelamos las estadísticas
            model.eval()
        # remplazo de última capa
        model.fc = nn.Linear(1024, num_classes)
        return model


    # ### 3.3 Ciclio de entrenamiento
    #
    # En este caso estamos aprovechando que ShuffleNet V2 no tiene capas de deserción y solo tiene una capa de clasificación para simplificar el ciclo de entrenamiento.

    # In[87]:


    def train_epoch(dl, model, opt, device):
        """Entrena una época"""
        # entrenamiento de una época
        for batch in dl:
            x = batch['x'].to(device)
            y_true = batch['y'].to(device)
            # computamos logits
            y_lgts = model(x)
            # computamos la pérdida
            loss = F.cross_entropy(y_lgts, y_true.long())
            # vaciamos los gradientes
            opt.zero_grad()
            # retropropagamos
            loss.backward()
            # actualizamos parámetros
            opt.step()


    def eval_epoch(dl, model, device, num_batches=None):
        """Evalua una época"""
        # evitamos que se registren las operaciones
        # en la gráfica de cómputo
        with torch.no_grad():

            losses, accs = [], []
            # validación de la época con num_batches
            # si num_batches==None, se usan todos los lotes
            for batch in take(dl, num_batches):
                x = batch['x'].to(device)
                y_true = batch['y'].to(device)
                # hacemos inferencia para obtener los logits
                y_lgts = model(x)
                # computamos las probabilidades
                y_prob = F.softmax(y_lgts, 1)
                # obtenemos la clase predicha
                y_pred = torch.argmax(y_prob, 1)

                # computamos la pérdida
                loss = F.cross_entropy(y_lgts, y_true.long())
                # computamos la exactitud
                acc = (y_true == y_pred).type(torch.float32).mean()

                # guardamos históricos
                losses.append(loss.item())
                accs.append(acc.item())

            loss = np.mean(losses) * 100
            acc = np.mean(accs) * 100

            return loss, acc


    def save_check_point(model, epoch, run_dir):
        """Guarda un punto de control."""
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            },
            join(run_dir, 'weights.pth')
        )


    def train(model, trn_dl, val_dl,
              trn_writer, val_writer, epochs,
              trn_batches=None, val_batches=None):

        # optimizador
        opt = optim.Adam(model.parameters(), lr=1e-3)

        # usamos GPU si está disponible
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # movemos a dispositivo
        model.to(device)

        # modelo en modo de evaluación
        # para transferencia de conocimiento además de congelar los pesos,
        # es importante congelar las estadísticas σ y μ
        model.eval()

        # ciclo de entrenamiento
        best_acc = 0
        for epoch in tqdm(range(epochs)):


            # entrenamos la época
            train_epoch(trn_dl, model, opt, device)

            # evaluamos la época en entrenamiento
            trn_loss, trn_acc = eval_epoch(trn_dl, model, device, trn_batches)

            # registramos trazas de TB
            trn_writer.add_scalar('metrics/loss', trn_loss, epoch)
            trn_writer.add_scalar('metrics/acc', trn_acc, epoch)

            # evaluamos la época en validación
            val_loss, val_acc = eval_epoch(val_dl, model, device, val_batches)

            # registramos trazas de TB
            val_writer.add_scalar('metrics/loss', val_loss, epoch)
            val_writer.add_scalar('metrics/acc', val_acc, epoch)


            # pseudo paro temprano: si hay mejora guardamos punto de control
            if val_acc > best_acc:
                best_acc = val_acc
                save_check_point(model, epoch, run_dir)


    # ### 3.4 Entrenamiento y monitoreo

    # In[88]:


    # directorio de la corrida
    run_dir = join('runs', 'flowers17', timestamp())
    run_dir


    # In[89]:


    # escritor de trazas
    trn_writer = SummaryWriter(join(run_dir, 'trn'))
    val_writer = SummaryWriter(join(run_dir, 'val'))


    # In[90]:

    #
    # # inspeccionemos el directorio de la corrida
    # get_ipython().system('ls -R {run_dir}')
    #
    #
    # # In[91]:
    #
    #
    # # lanzamos Tensorboard
    # get_ipython().run_line_magic('load_ext', 'tensorboard')
    # get_ipython().run_line_magic('tensorboard', '--logdir runs/flowers17 --host localhost')


    # In[92]:


    # instanciamos modelo con pesos
    model = shufflenet_v2_x0_5()
    # entrenamos modelo
    train(model, trn_dl, val_dl, trn_writer, val_writer,
          epochs=20, trn_batches=5, val_batches=5)


    # In[ ]:


    # inspeccionemos el directorio de la corrida
    # get_ipython().system('ls -R {run_dir}')


    # ## 4 Evaluación

    # ### 4.1 Cargando modelo

    # In[ ]:


    # cargamos el punto de contral
    ckpt = torch.load(join(run_dir, 'weights.pth'))
    # imprimimos la mejor época
    best_epoch = ckpt['epoch']
    print(f'Best epoch {best_epoch}')


    # In[ ]:


    # instanciamos un modelo
    model = shufflenet_v2_x0_5()

    # cargamos pesos
    state_dict = ckpt['model_state_dict']
    model.load_state_dict(state_dict)

    # congelamos las estadísticas
    model.eval()

    epoch = ckpt['epoch']
    print(f'Cargamos el mejor modelo, época {epoch}.')


    # ### 4.2 Conjunto y cargador de prueba

    # In[ ]:


    # conjunto
    tst_ds = Flowers17(DATA_DIR, 1, 'tst', transform=val_tsfm)
    # cargador
    tst_dl = DataLoader(
        # conjunto
        tst_ds,
        # tamaño del lote
        batch_size=BATCH_SIZE,
        # desordenar
        shuffle=True,
        # número de procesos paralelos
        num_workers=0
    )
    len(tst_ds)


    # ### 4.3 Evaluación Final

    # In[ ]:


    device = torch.device('cpu')
    trn_loss, trn_acc = eval_epoch(trn_dl, model, device)
    val_loss, val_acc = eval_epoch(val_dl, model, device)
    tst_loss, tst_acc = eval_epoch(tst_dl, model, device)


    # In[ ]:


    print(f'ac trn={trn_acc:5.2f} val={val_acc:5.2f} tst={tst_acc:5.2f}')


    # In[ ]:


    print(f'loss trn={trn_loss:6.2f} val={val_loss:6.2f} tst={tst_loss:6.2f}')


    # ### 4.4 Inspección visual de resultados

    # In[ ]:


    with torch.no_grad():
        batch = next(iter(tst_dl))
        x, y_true = batch['x'], batch['y']
        y_pred = torch.argmax(F.softmax(model(x), 1), 1)
        titles = [f'V={t} P={p}' for t, p in zip(y_true, y_pred)]
        display_batch(x, titles, ROWS, COLS)


    # ## 5. Participación
    #
    # Enriquece el proceso de aumentado de datos agregando un corte aleatorio. Para esto, modifica la transformación de entrenamiento realizando los siguientes pasos:
    #
    # 1. Modifica la línea `T.Resize(IMG_SIZE)` para que la imagen redimensionada sea 1.25 veces `IMG_SIZE`.
    # 2. Enseguida agrega un corte aletorio usando la trasformación [`RandomCrop`](https://pytorch.org/docs/1.6.0/torchvision/transforms.html#torchvision.transforms.RandomCrop).

    # ## 6. Participación
    #
    # Descomenta las líneas de código en la siguiente celda y correla.
    #
    # * ¿Qué pasa con las evaluaciones finales?
    # * ¿Por qué crees que se da este comportamiento?

    # In[ ]:


    # model = shufflenet_v2_x0_5(freeze_params_buffers=False)

    # state_dict = ckpt['model_state_dict']
    # model.load_state_dict(state_dict)

    # trn_loss, trn_acc = eval_epoch(trn_dl, model, device)
    # val_loss, val_acc = eval_epoch(val_dl, model, device)
    # tst_loss, tst_acc = eval_epoch(tst_dl, model, device)

    # print(f'acc trn={trn_acc:5.2f} val={val_acc:5.2f} tst={tst_acc:5.2f}')
    # print(f'loss trn={trn_loss:6.2f} val={val_loss:6.2f} tst={tst_loss:6.2f}')

