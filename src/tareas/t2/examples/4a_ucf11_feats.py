#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/gibranfp/CursoAprendizajeProfundo/blob/2022-1/notebooks/4a_ucf11_feats.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Extracción de características convolucionales de cuadros de video
# 
# Curso: [Aprendizaje Profundo](http://turing.iimas.unam.mx/~gibranfp/cursos/aprendizaje_profundo/). Profesor: [Gibran Fuentes Pineda](http://turing.iimas.unam.mx/~gibranfp/). Ayudantes: [Bere](https://turing.iimas.unam.mx/~bereml/) y [Ricardo](https://turing.iimas.unam.mx/~ricardoml/).
# 
# ---
# ---
# 
# En esta libreta usaremos un modelo CNN preentreando como extractor de características convolucionales de cuadros del conjunto [UCF11](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php).
# 
# <img src="https://www.cs.ucf.edu/~liujg/realistic_actions/youtube_snaps.jpg" />

# ## 1 Preparación

# In[1]:


# sistema de archivos
import os
# listar archivos por patrón
from glob import glob
# sistema de archivos
from os.path import join
# flush!
import sys

# arreglos multidimensionales
import numpy as np
# redes neuronales
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
# almacenamiento de arreglos multidimensionales
import zarr
# redes
from torch.utils.data import DataLoader
from torchvision.io import read_video
# barras de progreso
from tqdm.auto import tqdm

DATA_DIR = os.path.join('..', 'data')

BATCH_SIZE = 5
# numéro de cuadros por video
NUM_FRAMES = 10
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

if __name__ == '__main__':

    #%%
    class UCF11:
        def __init__(self, videos_dir, num_frames, tsfm):

            pattern = join(videos_dir, '*', '*', '*.mpg')
            self.paths = sorted(glob(pattern))
            self.num_frames = num_frames
            self.tsfm = tsfm
            # removemos videos demasiado cortos
            self._filter_out_videos()
            # UCF11 classes
            classes = (
                'basketball', 'biking', 'diving', 'golf_swing',
                'horse_riding', 'soccer_juggling', 'swing', 'tennis_swing',
                'trampoline_jumping', 'volleyball_spiking', 'walking'
            )
            self.cls_idx = {c: i for i, c in enumerate(classes)}

        def _filter_out_videos(self):
            """Remueve videos con menos de `num_frames` frames."""
            print(f'Removiendo videos con menos de {self.num_frames} cuadros')
            sys.stdout.flush()
            too_short = []
            for path in tqdm(self.paths):
                frames = read_video(path, pts_unit='sec')[0].shape[0]
                if frames < self.num_frames:
                    too_short.append(path)
            for path in too_short:
                self.paths.remove(path)

        def __getitem__(self, i):
            path = self.paths[i]

            # ubtenemos subruta class/group/video.mpg
            parts = path.split('/')[4:]
            subpath = '-'.join([parts[0], parts[2]])[:-4]

            # leemos el video completo
            frames = read_video(path, pts_unit='sec')[0]
            # calculamos el salto
            step = frames.shape[0] // self.num_frames
            # creamos indices saltando
            indices = np.arange(0, step * self.num_frames, step)
            # calculamos los indices restantes y dividismo entre 2
            offset = (frames.shape[0] - indices[-1]) // 2
            # recorremos a la derecha para centrar
            indices += offset
            # seleccionamos los cuadros
            frames = frames[indices]
            # convert to channel first
            frames = frames.movedim(3, 1)
            # aplicamos trasformación
            frames = self.tsfm(frames)

            # obtenemos etiqueta
            y = self.cls_idx[parts[0]]

            return subpath, frames, y

        def __len__(self):
            return len(self.paths)

    tsfm = T.Compose([
        # redimensionamos a 224x224
        T.Resize(IMG_SIZE),
        # cortamos al centro
        T.CenterCrop(IMG_SIZE),
        # uint => float, x ∈ [0, 1]
        T.ConvertImageDtype(torch.float),
        # estandarizamos con media y desviación estandar
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


    # ## 3 Extracción

    # Se asume que [UCF11](https://www.crcv.ucf.edu/data/UCF11_updated_mpg.rar) se descargó y extrajó en `DATA_DIR/ucf11`.


    data_dir = join(DATA_DIR, 'ucf11')
    zarr_dir = join(data_dir, 'ucf11.zarr')

    if not os.path.isdir(zarr_dir):

        videos_dir = join(data_dir, 'UCF11_updated_mpg')
        print(f'Usando {videos_dir}')

        ds = UCF11(videos_dir, NUM_FRAMES, tsfm)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)

        print(f'Extrayendo características en {zarr_dir}')
        sys.stdout.flush()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = tvm.densenet121(pretrained=True)
        model.classifier = nn.Identity()
        model.eval()
        model = model.to(device)

        z = zarr.open(zarr_dir, 'w')

        with torch.no_grad():

            for subpaths, frames, ys in tqdm(dl):

                # datos a dispositivo
                frames = frames.to(device)

                # guardamos dimensiones
                b, s, *img = frames.shape
                # planamos lote y secuencia en una sola dimensión
                frames = frames.reshape(-1, *img)
                # computamos características conv
                feats = model(frames)
                # restauramos lote y secuencia
                feats = feats.reshape(b, s, -1)
                # movemos a cpu y numpy
                feats = feats.cpu().numpy()

                # guardamos
                for subpath, x, y in zip(subpaths, feats, ys):
                    # creamos arreglo
                    arr = z.create_dataset(subpath, data=x, dtype=np.float32)
                    # asignamos etiqueta
                    arr.attrs['y'] = y.item()

    else:
        print(f'Características ya extraidas en {zarr_dir}')


