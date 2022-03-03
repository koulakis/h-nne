from enum import Enum

import numpy as np
import pandas as pd
import mat73
from gensim import models
from sklearn import preprocessing
from PIL import Image
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import pickle


def load_coil20(data_path):
    coil_data_path = data_path / 'coil-20-proc'

    # noinspection PyTypeChecker
    x_coil20, y_coil20 = zip(*[
        (np.array(Image.open(path)).flatten(), path.stem.split('__')[0])
        for path in coil_data_path.glob('*.png')
    ])

    x_coil20, y_coil20 = np.vstack(x_coil20), np.hstack(y_coil20)

    le = preprocessing.LabelEncoder()

    y_coil20 = y_coil20.tolist()
    y_coil20 = le.fit_transform(y_coil20)

    return x_coil20, y_coil20


def load_shuttle(data_path):
    shuttle = pd.read_csv(data_path / 'shuttle.tst', sep=' ', header=None)
    shuttle_train = pd.read_csv(data_path / 'shuttle_train.csv', index_col='Id')

    x_shuttle, y_shuttle = np.vstack([np.array(shuttle_train)[:, :-1], np.array(shuttle)[:, :-1]]), np.hstack(
        [shuttle_train['Category'], np.array(shuttle)[:, -1]])

    return x_shuttle, y_shuttle


def load_mnist(data_path):
    x_train_mnist, y_train_mnist = zip(*MNIST(data_path, train=True, download=True))
    x_test_mnist, y_test_mnist = zip(*MNIST(data_path, train=False, download=True))

    x_mnist = np.vstack([np.array(image).flatten() for image in x_train_mnist + x_test_mnist])
    y_mnist = np.hstack([y_train_mnist, y_test_mnist])

    return x_mnist, y_mnist


def load_fmnist(data_path):
    x_train_fmnist, y_train_fmnist = zip(*FashionMNIST(data_path, train=True, download=True))
    x_test_fmnist, y_test_fmnist = zip(*FashionMNIST(data_path, train=False, download=True))

    x_fmnist = np.vstack([np.array(image).flatten() for image in x_train_fmnist + x_test_fmnist])
    y_fmnist = np.hstack([y_train_fmnist, y_test_fmnist])

    return x_fmnist, y_fmnist


def load_imagenet_test(data_path):
    imagenet_x = mat73.loadmat(data_path / 'imagenet_ILSVRC_2012/val/data.mat')['data']
    imagenet_y = mat73.loadmat(data_path / 'imagenet_ILSVRC_2012/val/labels.mat')['labels']

    return imagenet_x, imagenet_y


def load_cifar_10(data_path):
    x_train_cifar10, y_train_cifar10 = zip(*CIFAR10(data_path, train=True, download=True))
    x_test_cifar10, y_test_cifar10 = zip(*CIFAR10(data_path, train=False, download=True))

    x_cifar10 = np.vstack([np.array(image).flatten() for image in x_train_cifar10 + x_test_cifar10])
    y_cifar10 = np.hstack([y_train_cifar10, y_test_cifar10])

    return x_cifar10, y_cifar10


def load_bbt(data_path):
    bbt = np.load(data_path / 'bbt_data.npy')
    return bbt[:, :-1], bbt[:, -1]


def load_buffy(data_path):
    buffy = np.load(data_path / 'buffy_data.npy')
    return buffy[:, :-1], buffy[:, -1]


def load_imagenet_train(data_path):
    imagenet_x = mat73.loadmat(data_path / 'imagenet_ILSVRC_2012/train/data.mat')['data']
    imagenet_y = mat73.loadmat(data_path / 'imagenet_ILSVRC_2012/train/labels.mat')['labels']

    return imagenet_x, imagenet_y


def load_google_news(data_path):
    data = models.KeyedVectors.load_word2vec_format(
        data_path / 'google-news/GoogleNews-vectors-negative300.bin',
        binary=True
    ).vectors
    targets = np.zeros(data.shape[0], dtype=int)

    return data, targets


def load_higgs(data_path):
    file_name = data_path / 'HIGGS.csv'
    with open(file_name, "rb") as f:
        # noinspection PyTypeChecker
        mat = np.loadtxt(f, delimiter=",")
    return mat[:, 1:], mat[:, 0]


def load_mnist_8m(data_path):
    mnist = np.load(data_path / 'mnist8M/data.npy')
    return mnist[:, :-1], mnist[:, -1]


small_datasets = {
    'coil_20': load_coil20,
    'shuttle': load_shuttle,
    'mnist': load_mnist,
    'fmnist': load_fmnist,
}

medium_datasets = {
    'imagenet_test': load_imagenet_test,
    'cifar_10': load_cifar_10,
    'bbt': load_bbt,
    'buffy': load_buffy
}

large_datasets = {
    'imagenet_train': load_imagenet_train,
    'google_news': load_google_news,
    'higgs': load_higgs,
    'mnist_8m': load_mnist_8m
}


def get_dataset_loader(datasets):
    return {
        dataset_name: dataset
        for dataset_name, dataset in datasets.items()
    }


class DatasetGroup(str, Enum):
    small = 'small'
    medium = 'medium'
    large = 'large'


def dataset_loaders(dataset_group):
    mapping = {
        DatasetGroup.small: small_datasets,
        DatasetGroup.medium: medium_datasets,
        DatasetGroup.large: large_datasets,
    }

    return get_dataset_loader(mapping[dataset_group])


validation_1nn = [1]
validation_ranges_small = [1, 10, 20, 40]
validation_ranges_large = [100, 200]

dataset_validation_knn_values = {
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
        finch_info['partition_clustering']
    )
