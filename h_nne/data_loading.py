from pathlib import Path

import numpy as np
import pandas as pd
import mat73
import sklearn.datasets
from gensim import models
from sklearn import preprocessing
from PIL import Image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import pickle

PYTORCH_DATASETS_PATH = 'dimensionality_reduction'

COIL_20_PATH = '/cvhci/data/dimensionality_reduction/coil-20-proc'
SHUTTLE_PATH = '/cvhci/data/dimensionality_reduction/shuttle.tst'
SHUTTLE_TRAIN_PATH = '/cvhci/data/dimensionality_reduction/shuttle_train.csv'
BBT_PATH = '/cvhci/data/dimensionality_reduction/bbt_data.npy'
BUFFY_PATH = '/cvhci/data/dimensionality_reduction/buffy_data.npy'
MNIST8M_PATH = '/cvhci/data/dimensionality_reduction/mnist8M/data.npy'
MNIST2M_PATH = '/cvhci/data/dimensionality_reduction/mnist2M/data.npy'
MNIST4M_PATH = '/cvhci/data/dimensionality_reduction/mnist4M/data.npy'


# Small datasets
def load_pendigits():
    pendigits = sklearn.datasets.load_digits()
    return pendigits.data, pendigits.target


def load_coil20():
    coil_data_path = Path(COIL_20_PATH)

    X_coil20, y_coil20 = zip(*[
        (np.array(Image.open(path)).flatten(), path.stem.split('__')[0])
        for path in coil_data_path.glob('*.png')
    ])

    X_coil20, y_coil20 = np.vstack(X_coil20), np.hstack(y_coil20)
    
    le = preprocessing.LabelEncoder()
    
    y_coil20 = y_coil20.tolist()
    y_coil20 = le.fit_transform(y_coil20)
    
    return X_coil20, y_coil20


def load_shuttle():
    shuttle = pd.read_csv(SHUTTLE_PATH, sep=' ', header=None)
    shuttle_train = pd.read_csv(SHUTTLE_TRAIN_PATH, index_col='Id')

    X_shuttle, y_shuttle = np.vstack([np.array(shuttle_train)[:, :-1], np.array(shuttle)[:, :-1]]), np.hstack([shuttle_train['Category'], np.array(shuttle)[:, -1]])
    
    return X_shuttle, y_shuttle


def load_mnist():
    X_train_MNIST, y_train_MNIST = zip(*MNIST(PYTORCH_DATASETS_PATH, train=True, download=True))
    X_test_MNIST, y_test_MNIST = zip(*MNIST(PYTORCH_DATASETS_PATH, train=False, download=True))

    X_MNIST = np.vstack([np.array(image).flatten() for image in X_train_MNIST + X_test_MNIST])
    y_MNIST = np.hstack([y_train_MNIST, y_test_MNIST])
    
    return X_MNIST, y_MNIST


def load_fmnist():
    X_train_FMNIST, y_train_FMNIST = zip(*FashionMNIST(PYTORCH_DATASETS_PATH, train=True, download=True))
    X_test_FMNIST, y_test_FMNIST = zip(*FashionMNIST(PYTORCH_DATASETS_PATH, train=False, download=True))

    X_FMNIST = np.vstack([np.array(image).flatten() for image in X_train_FMNIST + X_test_FMNIST])
    y_FMNIST = np.hstack([y_train_FMNIST, y_test_FMNIST])
    
    return X_FMNIST, y_FMNIST


# Medium datasets
def load_bbt():
    bbt = np.load(BBT_PATH)
    return bbt[:, :-1], bbt[:, -1]


def load_buffy():
    buffy = np.load(BUFFY_PATH)
    return buffy[:, :-1], buffy[:, -1]


def load_imagenet_test():
    imagenet_X = mat73.loadmat('/cvhci/data/dimensionality_reduction/imagenet_ILSVRC_2012/val/data.mat')['data']
    imagenet_y = mat73.loadmat('/cvhci/data/dimensionality_reduction/imagenet_ILSVRC_2012/val/labels.mat')['labels']
    
    return imagenet_X, imagenet_y


def load_imagenet_500K():
    imagenet = np.load('/cvhci/data/dimensionality_reduction/imagenet_500K/data.npy')
    
    return imagenet[:, :-1], imagenet[:, -1]
    

def load_imagenet_train():
    imagenet_X = mat73.loadmat('/cvhci/data/dimensionality_reduction/imagenet_ILSVRC_2012/train/data.mat')['data']
    imagenet_y = mat73.loadmat('/cvhci/data/dimensionality_reduction/imagenet_ILSVRC_2012/train/labels.mat')['labels']
    
    return imagenet_X, imagenet_y


def load_cifar_10():
    X_train_CIFAR10, y_train_CIFAR10 = zip(*CIFAR10(PYTORCH_DATASETS_PATH, train=True, download=True))
    X_test_CIFAR10, y_test_CIFAR10 = zip(*CIFAR10(PYTORCH_DATASETS_PATH, train=False, download=True))

    X_CIFAR10 = np.vstack([np.array(image).flatten() for image in X_train_CIFAR10 + X_test_CIFAR10])
    y_CIFAR10 = np.hstack([y_train_CIFAR10, y_test_CIFAR10])
    
    return X_CIFAR10, y_CIFAR10


# Large datasets
def load_google_news():
    data = models.KeyedVectors.load_word2vec_format(
        '/cvhci/data/dimensionality_reduction/google-news/GoogleNews-vectors-negative300.bin', 
        binary=True
    ).vectors
    targets = np.zeros(data.shape[0], dtype=int)
    
    return data, targets


def load_mnist_8m():
    mnist = np.load(MNIST8M_PATH)
    return mnist[:, :-1], mnist[:, -1]


def load_mnist_4m():
    mnist = np.load(MNIST4M_PATH)
    return mnist[:, :-1], mnist[:, -1]


def load_mnist_2m():
    mnist = np.load(MNIST2M_PATH)
    return mnist[:, :-1], mnist[:, -1]


def load_higgs():
    file_name = '/cvhci/data/FaceClust/Higgs/HIGGS.csv'
    mat = np.loadtxt(open(file_name, "rb"), delimiter=",")
    return mat[:, 1:], mat[:, 0]


dataset_loaders = {
    'pendigits': load_pendigits,
    'coil_20': load_coil20, 
    'shuttle': load_shuttle, 
    'mnist': load_mnist,
    'fmnist': load_fmnist, 
    'imagenet_test': load_imagenet_test,
    'cifar_10': load_cifar_10,
    'bbt': load_bbt,
    'buffy': load_buffy
}

large_dataset_loaders = {
    'imagenet_train': load_imagenet_train,
    'google_news': load_google_news,
#     'imagenet_500K': load_imagenet_500K,
#     'mnist_2m': load_mnist_2m,
#     'mnist_4m': load_mnist_4m,
    'higgs': load_higgs,
    'mnist_8m': load_mnist_8m
}

validation_1nn = [1]
validation_ranges_small = [1, 10, 20, 40]
validation_ranges_large = [100, 200] #, 400, 800, 1600, 3200]


dataset_validation_knn_values = {
    'pendigits': validation_ranges_small,
    'coil_20': validation_ranges_small,
    'shuttle': validation_ranges_small,
    'mnist': validation_ranges_small,
    'fmnist': validation_ranges_small,
    'bbt': validation_ranges_small,
    'buffy': validation_ranges_small,
    'imagenet_test': validation_ranges_small,
    'google_news': validation_ranges_small,
    'mnist_8m': validation_ranges_small,
    'imagenet_train': validation_ranges_small
}

def load_extracted_finch_partitions(path):
    with open(path, "rb") as f:
         finch_info = pickle.load(f)
    
    return (
            finch_info['partitions'], 
            finch_info['partition_sizes'], 
            finch_info['req_c'], 
            finch_info['adjacency_matrices'], 
            finch_info['partition_clustering'],
            finch_info['cluster_dists'],
            finch_info['first_neighbors_list']
    )
