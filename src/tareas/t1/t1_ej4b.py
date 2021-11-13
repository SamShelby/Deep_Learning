import random
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import islice as take
import tensorflow as tf
import tensorboard as tb
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import seaborn as sns
sns.set_theme()
sns.set_context("poster")

activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}


class DenseNet(nn.Module):
    def __init__(self, n_hidden_layers, n_nodes, activation='relu'):
        super().__init__()
        self.activation = activation
        self.image_flatten_dim = 1 * 28 * 28
        self.flatten = nn.Flatten()
        self.in_layer = nn.Linear(in_features=self.image_flatten_dim, out_features=n_nodes)
        self.fcs = [nn.Linear(in_features=n_nodes, out_features=n_nodes) for i in range(n_hidden_layers - 1)]
        self.out_layer = nn.Linear(in_features=n_nodes, out_features=10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.in_layer(x)
        for layer in self.fcs:
            x = activations[self.activation](layer(x))
        x = self.out_layer(x)
        return x


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_epoch(dataloader, model, optimizer, criterion):
    # por cada lote
    for inputs, labels in dataloader:
        # computamos logits
        outputs = model(inputs)

        # computamos la pérdida
        loss = criterion(outputs, labels)

        # vaciamos los gradientes
        optimizer.zero_grad()

        # retropropagamos
        loss.backward()

        # actualizamos parámetros
        optimizer.step()


def eval_epoch(dataloader, model, criterion, num_batches=None):
    # evitamos que se registren las operaciones
    # en la gráfica de cómputo
    with torch.no_grad():
        # historiales
        losses = []

        # validación de la época con num_batches
        # si num_batches==None, se usan todos los lotes
        for inputs, labels in take(dataloader, num_batches):
            # computamos los logits
            outputs = model(inputs)

            # computamos la pérdida
            loss = criterion(outputs, labels)

            # guardamos históricos
            losses.append(loss.item())

        # promediamos
        loss = np.mean(losses)

        return loss


def train(model, optimizer, train_dl, test_dl, criterion, epochs=20,
          trn_batches=None, tst_batches=None, verbose=1):
    # historiales
    loss_hist = []

    # ciclo de entrenamiento
    for epoch in range(epochs):
        # entrenamos la época
        train_epoch(train_dl, model, optimizer, criterion)

        # evaluamos la época en entrenamiento
        trn_loss = eval_epoch(train_dl, model, criterion, trn_batches)
        # evaluamos la época en prueba
        tst_loss = eval_epoch(test_dl, model, criterion, tst_batches)

        # guardamos historial
        loss_hist.append([trn_loss, tst_loss])

        # imprimimos progreso
        if verbose > 0:
            print(f' E{epoch:02} '
                  f'loss=[{trn_loss:6.2f},{tst_loss:6.2f}] ')

    return np.array(loss_hist)


def plot_loss_hist(loss_hist, title=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(loss_hist[:, 0], label='Train Loss')
    ax.plot(loss_hist[:, 1], label='Test Loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('epoch')
    plt.xlabel('loss')
    plt.show()


def train_test_model(model_cfg, plot_=False, verbose=1, model_arch='Dense'):
    if model_arch == 'Dense':
        model = DenseNet(n_hidden_layers=model_cfg['n_hidden_layers'], n_nodes=model_cfg['n_nodes'],
                         activation=model_cfg['activation'])
    else:
        model = CnnNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=model_cfg['learning_rate'], momentum=0.9)
    t0 = time.time()
    loss_hist = train(model, optimizer, trainloader, testloader, criterion,
                      epochs=model_cfg['n_epochs'], verbose=verbose)
    if verbose > 0:
        print('Train Time: {} s'.format(round(time.time() - t0, 4)))
        print('Test Loss: {}'.format(round(loss_hist[-1][-1], 2)))
    if plot_:
        plot_loss_hist(loss_hist, str(model_cfg))
    return model, loss_hist


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def calc_probs(dataloader, model):
    class_probs = []
    class_label = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            output = model(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]

            class_probs.append(class_probs_batch)
            class_label.append(labels)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_label = torch.cat(class_label)

    return test_probs, test_label


def compute_curve(labels, predictions, num_thresholds=127, weights=None):
    _MINIMUM_COUNT = 1e-7

    if weights is None:
        weights = 1.0

    # Compute bins of true positives and false positives.
    bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
    float_labels = labels.astype(np.float64)
    histogram_range = (0, num_thresholds - 1)
    tp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=float_labels * weights)
    fp_buckets, _ = np.histogram(
        bucket_indices,
        bins=num_thresholds,
        range=histogram_range,
        weights=(1.0 - float_labels) * weights)

    # Obtain the reverse cumulative sum.
    tp = np.cumsum(tp_buckets[::-1])[::-1]
    fp = np.cumsum(fp_buckets[::-1])[::-1]
    tn = fp[0] - fp
    fn = tp[0] - tp
    precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)
    recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)
    return np.stack((tp, fp, tn, fn, precision, recall))


def make_np(x):
    if isinstance(x, np.ndarray):
        return x
    if np.isscalar(x):
        return np.array([x])
    if isinstance(x, torch.Tensor):
        if isinstance(x, torch.autograd.Variable):
            x = x.data
        x = x.cpu().numpy()
        return x
    raise NotImplementedError(
        'Got {}, but numpy array, torch tensor, or caffe2 blob name are expected.'.format(type(x)))


def pr_curve(class_index, test_probs, test_label):
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]
    return compute_curve(make_np(tensorboard_truth), make_np(tensorboard_probs))


def plot_pr_curves(n_classes, test_probs, test_label):
    fig, axes = plt.subplots(2, 5, figsize=(20, 15))
    for i in range(n_classes):
        curve = pr_curve(i, test_probs, test_label)
        axes[i % 2][i // 2].plot(curve[4, :-1], curve[5, :-1])
        axes[i % 2][i // 2].title.set_text('Class = {}'.format(i))
        axes[i % 2][i // 2].set_xlabel('recall')
        axes[i % 2][i // 2].set_ylabel('precision')
    plt.suptitle('Precision-Recall Curves')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    # %%
    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    # datasets and dataloaders
    trainset = FashionMNIST('./data', download=True, train=True, transform=transform)
    testset = FashionMNIST('./data', download=True, train=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # %%
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # create grid and show images
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)

    # %%
    model_cfg = {'n_hidden_layers': 2, 'n_nodes': 100, 'activation': 'sigmoid',
                'n_epochs': 5, 'learning_rate': 0.001}
    model, loss_hist = train_test_model(model_cfg, plot_=True)

    # obtenemos la probabilidad de las predicciones en un tensor de: test_size x num_classes
    test_probs, test_label = calc_probs(testloader, model)

    # graficamos todas las curvas de precisión-recall
    plot_pr_curves(len(classes), test_probs, test_label)

    # %%
    model_cfg = {'n_hidden_layers': 1, 'n_nodes': 200, 'activation': 'sigmoid',
                 'n_epochs': 5, 'learning_rate': 0.001}
    model, loss_hist = train_test_model(model_cfg, plot_=True)

    # obtenemos la probabilidad de las predicciones en un tensor de: test_size x num_classes
    test_probs, test_label = calc_probs(testloader, model)

    # graficamos todas las curvas de precisión-recall
    plot_pr_curves(len(classes), test_probs, test_label)

    # %%
    model_cfg = {'n_hidden_layers': 3, 'n_nodes': 100, 'activation': 'relu',
                 'n_epochs': 8, 'learning_rate': 0.001}
    model, loss_hist = train_test_model(model_cfg, plot_=True)

    # obtenemos la probabilidad de las predicciones en un tensor de: test_size x num_classes
    test_probs, test_label = calc_probs(testloader, model)

    # graficamos todas las curvas de precisión-recall
    plot_pr_curves(len(classes), test_probs, test_label)

    # %%
    model_cfg = {'n_epochs': 12, 'learning_rate': 0.001}
    model, loss_hist = train_test_model(model_cfg, plot_=True, model_arch='Cnn')

    # obtenemos la probabilidad de las predicciones en un tensor de: test_size x num_classes
    test_probs, test_label = calc_probs(testloader, model)

    # graficamos todas las curvas de precisión-recall
    plot_pr_curves(len(classes), test_probs, test_label)
