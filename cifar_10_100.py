#!/usr/bin/env python3
#
# cifar_10_100.py
#
#     https://github.com/JulianChia/cifar_10_100
#
# Copyright (C) 2022 Julian Chia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__all__ = ['load_CIFAR']

from pathlib import Path
from typing import List, Tuple, Any
from urllib.request import build_opener, install_opener, urlretrieve
import numpy as np
from dataklasses import dataklass


@dataklass
class CIFARimages:
    batch_label: List
    filenames: List
    nimages: int
    nchannels: int
    nrows: int
    ncols: int
    pixels: np.array


@dataklass
class CIFAR10labels:
    batch_label: List
    nlabels: int
    labels_str: Tuple
    labels: np.array


@dataklass
class CIFAR100labels:
    batch_label: List
    nfine_labels: int
    ncoarse_labels: int
    fine_labels_str: Tuple
    coarse_labels_str: Tuple
    fine_labels: np.array
    coarse_labels: np.array


def load_CIFAR(db, path=None, normalise=True, flatten=True, onehot=True):
    """Function to download and extract the CIFAR10 or CIFAR100 datasets into
    dataklass objections for deep learning.

    dataklass from https://github.com/dabeaz/dataklasses

    Args:
     db - int : Dataset annotation. Accepted value is either 10 or 100 to
                refer to CIFAR-10 or CIFAR-100 datasets.

    Kwargs:
     path - str: CIFAR datasets directory. Default to current directory/CIFAR.
                 Create if nonexistant. Download any missing CIFAR files.
     normalise - boolean: yes -> pixels RGB values [0,255] divided by 255.
                          no  -> pixels RGB values [0,255].
     flatten   - boolean: yes -> pixels of each image stored as 1D numpy array.
                          no  -> pixels of each image stored as 3D numpy array.
     onehot    - boolean: yes -> labels stored as one-hot encoded numpy array.
                          no  -> labels values used.

    Returns:
     {'train': {'images': train_images, 'labels': train_labels},
      'test': {'images': test_images, 'labels': test_labels}}
     where,
      train_images = CIFARimages(batch_label=[byte strings],
                                 filenames=[byte strings],
                                 nimages=50000,
                                 nchannels=3, nrows=32, ncols=32,
                                 pixels=np.array())
            if normalise, pixels dtype='float64'
            else,         pixels dtype='uint8'
            if flatten,   pixels.shape=(50000, 3072)
            else,         pixels.shape=(50000, 3, 32, 32)
      train_labels = CIFAR10labels(batch_label=[byte strings],
                                   nlabels=50000,
                                   labels_str=() with 10 byte-str,
                                   labels=np.array() dtype='uint8')
            if onehot,    labels.shape=(50000, 10)
            else,         labels.shape=(50000)
      train_labels = CIFAR100labels(batch_label=[byte strings],
                                    nlabels=50000,
                                    coarse_labels_str=() with 20 byte-str,
                                    fine_labels_str=() with 100 byte-str,
                                    coarse_labels=np.array() dtype='uint8',
                                    fine_labels=np.array() dtype='uint8')
            if onehot,    coarse_labels.shape=(50000, 20)
                          fine_labels.shape=(50000, 100)
            else,         coarse_labels.shape=(50000)
                          fine_labels.shape=(50000)
     test_images = CIFARimages(batch_label=[byte strings],
                               filenames=[byte strings],
                               nimages=50000,
                               nchannels=3, nrows=32, ncols=32,
                               pixels=np.array())
            if normalise, pixels dtype='float64'
            else,         pixels dtype='uint8'
            if flatten,   pixels.shape=(10000, 3072)
            else,         pixels.shape=(10000, 3, 32, 32)
     test_labels = CIFAR10labels(batch_label=[byte strings],
                                 nlabels=10000,
                                 labels_str=() with 10 byte-str,
                                 labels=np.array() dtype='uint8')
            if onehot,    labels.shape=(10000, 10)
            else,         labels.shape=(10000)
     test_labels = CIFAR100labels(batch_label=[byte strings],
                                  nlabels=10000,
                                  coarse_labels_str=() with 20 byte-str,
                                  fine_labels_str=() with 100 byte-str,
                                  coarse_labels=np.array() dtype='uint8',
                                  fine_labels=np.array() dtype='uint8')
            if onehot,    coarse_labels.shape=(10000, 20)
                          fine_labels.shape=(10000, 100)
            else,         coarse_labels.shape=(10000)
                          fine_labels.shape=(10000)
    """
    def _unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def _set_db():
        db_types = [10, 100]
        if db in db_types:
            if db == 10:
                return 'cifar10'
            else:
                return 'cifar100'
        else:
            raise ValueError('Unknown db. It must be "10" or "100".')

    def _set_CIFAR_dir():
        if not path:  # Set dir to current-directory/CIFAR
            return Path(__file__).parent.absolute() / 'CIFAR'
        else:  # Set dir to given path/CIFAR
            return Path(path) / 'CIFAR'

    def _download_CIFAR_dataset(datasets):
        """Download any missing files."""
        filename = datasets[db][0]
        url = datasets[db][1]
        filepath = path / filename
        if not filepath.exists():
            print(f'Downloading {filename} to {path}... ', end='')
            opener = build_opener()
            install_opener(opener)
            urlretrieve(url, filepath)
            print('Completed!')
        else:
            print(f'{filename} exists. No need to download.')

    def _extract_tar_gz_files(file, extractpath):
        import tarfile
        with tarfile.open(file, 'r:gz') as f:
            f.extractall(extractpath)

    def _onehot_encoding(labels):
        """Return a 2D numpy array where only the element for the correct label
        is 1 and other elements are 0.

        Args:
         labels - 1D np.array : CIFAR labels
        """
        rows = labels.size
        cols = labels.max() + 1
        onehot = np.zeros((rows, cols), dtype='uint8')
        onehot[np.arange(rows), labels] = 1
        return onehot

    def _get_cifar10_labels_images(batchfile):
        # 1. Get dataset
        dataset = _unpickle(batchfile)
        print(dataset.keys())
        # 2. Get dataset labels
        if db in ['cifar10']:
            lbs = np.array(dataset[b'labels'])  # if not onehot
        elif db in ['cifar100']:
            lbs = np.array(dataset[b'fine_labels'])  # if not onehot
        if onehot:
            lb = _onehot_encoding(lbs)
        lbs = CIFAR10labels([dataset[b'batch_label']], len(lbs),
                            cifar10_labels_str, lbs)
        # 3. Get dataset images
        imgs = dataset[b'data']
        if normalise:
            imgs = imgs / 255
        if not flatten:
            imgs = imgs.reshape((len(imgs), 3, 32, 32))
        imgs = CIFARimages([dataset[b'batch_label']], dataset[b'filenames'],
                           len(imgs), 3, 32, 32, imgs)
        return lbs, imgs

    def _get_cifar100_labels_images(batchfile):
        # 1. Get dataset
        dataset = _unpickle(batchfile)
        # notes: dataset keywrords are b'filenames', b'batch_label',
        # b'fine_labels', b'coarse_labels'& b'data'.
        # 2. Get dataset labels
        finelbs = np.array(dataset[b'fine_labels'])  # if not onehot
        coarselbs = np.array(dataset[b'coarse_labels'])  # if not onehot
        if onehot:
            finelbs = _onehot_encoding(finelbs)
            coarselbs = _onehot_encoding(coarselbs)
        lbs = CIFAR100labels([dataset[b'batch_label']],
                             len(coarselbs),
                             len(finelbs),
                             cifar100_coarse_labels_str,
                             cifar100_fine_labels_str,
                             coarselbs,
                             finelbs)
        # 3. Get dataset images
        imgs = dataset[b'data']
        if normalise:
            imgs = imgs / 255
        if not flatten:
            imgs = imgs.reshape((len(imgs), 3, 32, 32))
        imgs = CIFARimages([dataset[b'batch_label']], dataset[b'filenames'],
                           len(imgs), 3, 32, 32, imgs)
        return lbs, imgs

    # 1. Define local args
    db = _set_db()            # db is redefined
    path = _set_CIFAR_dir()   # path is redefined
    url = 'http://www.cs.toronto.edu/~kriz/'
    cfiles = ('cifar-10-python.tar.gz', 'cifar-100-python.tar.gz')
    ucdirs = ('cifar-10-batches-py', 'cifar-100-python')
    datasets = {'cifar10': (cfiles[0], url+cfiles[0], path/ucdirs[0]),
                'cifar100': (cfiles[1], url+cfiles[1], path/ucdirs[1])}
    cifar10_batches = ('data_batch_1', 'data_batch_2', 'data_batch_3',
                       'data_batch_4', 'data_batch_5', 'test_batch')
    cifar10_labels_str = (
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')
    cifar100_batches = ('train', 'test')
    cifar100_coarse_labels_str = (
        'aquatic mammals', 'fish', 'flowers', 'food containers',
        'fruit and vegetables', 'household electrical devices',
        'household furniture', 'insects', 'large carnivores',
        'large man-made outdoor things', 'large natural outdoor scenes',
        'large omnivores and herbivores', 'medium-sized mammals',
        'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
        'trees', 'vehicles 1', 'vehicles 2')
    cifar100_fine_labels_str = (
        'beaver', 'dolphin', 'otter', 'seal', 'whale',
        'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
        'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
        'bottles', 'bowls', 'cans', 'cups', 'plates',
        'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
        'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
        'bed', 'chair', 'couch', 'table', 'wardrobe',
        'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
        'bear', 'leopard', 'lion', 'tiger', 'wolf',
        'bridge', 'castle', 'house', 'road', 'skyscraper',
        'cloud', 'forest', 'mountain', 'plain', 'sea',
        'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
        'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
        'crab', 'lobster', 'snail', 'spider', 'worm',
        'baby', 'boy', 'girl', 'man', 'woman',
        'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
        'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
        'maple', 'oak', 'palm', 'pine', 'willow',
        'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
        'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor')

    # 2. Create path dir if it doesn't exist and download CIFAR datasets there
    #    if they do not exist.
    try:
        path.mkdir(mode=0o777, parents=False, exist_ok=False)
    except FileExistsError:
        print(f'{path} exists. No need to create.')
    else:
        print(f'{path} is created.')
    finally:
        # Download any missing files
        _download_CIFAR_dataset(datasets)

    # 3. Extract compressed dataset batches/files to disk
    tar_gz_file = path / datasets[db][0]
    _extract_tar_gz_files(tar_gz_file, path)

    # 4. Configure the dictionary that is to be returned for CIFAR-10 datasets.
    if db in ['cifar10']:

        # 4.1 Get Training datasets from 5 different batches into 1 batch
        batchlabels: List = []
        labels: List = []
        pixels: List = []
        filenames: List = []
        for batch in cifar10_batches[:-1]:
            info = _unpickle(datasets[db][2] / batch)
            # Notes:
            # - info is a dict with keywords b'batch_label', b'labels',
            #   b'data' & b'filenames'.
            # - Value of b'labels' is a list of integers.
            # - Value of b'data' is a np.array() shape 10000x3072 with uint8
            #   values. Each row of the array stores a 32x32 colour image.
            #   The first 1024 entries contain the red channel values, the next
            #   1024 the green, and the final 1024 the blue.
            # - Value of b'filenames' is a list of byte strings.
            batchlabels.append(info[b'batch_label'])
            labels.extend(info[b'labels'])
            pixels.append(info[b'data'])
            filenames.extend(info[b'filenames'])

        # 4.2 Define training labels
        labels = np.array(labels)  # if not onehot
        if onehot:
            labels = _onehot_encoding(labels)
        train_labels = CIFAR10labels(batchlabels, len(labels),
                                  cifar10_labels_str, labels)

        # 4.3 Define training images
        pixels = np.concatenate(pixels)  # is not normalised and is flattened
        if normalise:
            pixels = pixels / 255
        if not flatten:
            pixels = pixels.reshape((len(pixels), 3, 32, 32))
        train_images = CIFARimages(batchlabels, filenames, len(pixels),
                                   3, 32, 32, pixels)

        # 4.4 Get Test datasets
        test_dataset_file = datasets[db][2] / cifar10_batches[5]
        test_labels, test_images = _get_cifar10_labels_images(test_dataset_file)

    elif db in ["cifar100"]:
        train_dataset_file = datasets[db][2] / cifar100_batches[0]
        train_labels, train_images = _get_cifar100_labels_images(train_dataset_file)
        test_dataset_file = datasets[db][2] / cifar100_batches[1]
        test_labels, test_images = _get_cifar100_labels_images(test_dataset_file)

    # 5. Store extracted training and test datasets into dictionary
    train = {'images': train_images, 'labels': train_labels}
    test = {'images': test_images, 'labels': test_labels}
    return {'train': train, 'test': test}


if __name__ == "__main__":
    print(f'cifar10')
    cifar10 = load_CIFAR(10,
                         path=None, normalise=True, flatten=True, onehot=False)
    # print(f'cifar10 = {cifar10}')
    print((cifar10['train']['images'].pixels.shape))
    print((cifar10['train']['labels'].labels.shape))
    print((cifar10['test']['images'].pixels.shape))
    print((cifar10['test']['labels'].labels.shape))

    print(f'cifar100')
    cifar100 = load_CIFAR(100,
                          path=None, normalise=True, flatten=True, onehot=False)
    # print(f'cifar100 = {cifar100}')
    print((cifar100['train']['images'].pixels.shape))
    print((cifar100['train']['labels'].fine_labels.shape))
    print((cifar100['train']['labels'].coarse_labels.shape))
    print((cifar100['test']['images'].pixels.shape))
    print((cifar100['test']['labels'].fine_labels.shape))
    print((cifar100['test']['labels'].coarse_labels.shape))
