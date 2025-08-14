#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import os

def noniid_dirichlet(dataset, num_users, alpha=1):

    # Assuming dataset.targets is a list or tensor of targets
    if hasattr(dataset.targets, "numpy"):
        targets = dataset.targets.numpy()
    else:
        targets = np.array(dataset.targets)

    num_classes = len(np.unique(targets))

    # Create an empty list for each user
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}

    # Indices for all the data points in the dataset
    all_idxs = np.arange(len(targets))

    # Initialize an array to keep track of assigned sample indices for each class
    classwise_idxs = {i: np.where(targets == i)[0] for i in range(num_classes)}

    # Sample user distribution for each class using Dirichlet distribution
    for c in range(num_classes):
        # Draw samples from a Dirichlet distribution
        dirichlet_dist = np.random.dirichlet(np.ones(num_users) * alpha, size=1)[0]

        # Get indices for the current class
        idxs = classwise_idxs[c]

        # Calculate number of samples per user for this class
        num_samples_per_user = np.round(dirichlet_dist * len(idxs)).astype(int)

        # Shuffle indices to ensure random distribution
        np.random.shuffle(idxs)

        # Assign samples to each user based on the sampled distribution
        start = 0
        for user, num_samples in enumerate(num_samples_per_user):
            end = start + num_samples
            dict_users[user] = np.concatenate((dict_users[user], idxs[start:end]), axis=0)
            start = end

    return dict_users

def mnist_noniid_dirichlet(dataset, num_users, alpha=1):

    # Number of classes in the dataset
    num_classes = len(np.unique(dataset.targets.numpy()))

    # Create an empty list for each user
    dict_users = {i: np.array([], dtype=np.int64) for i in range(num_users)}  # Ensure indices are int64

    # Indices for all the data points in the dataset
    all_idxs = np.arange(len(dataset.data))

    # Labels for each data point
    labels = dataset.targets.numpy()

    # Initialize an array to keep track of assigned sample indices for each class
    classwise_idxs = {i: np.where(labels == i)[0] for i in range(num_classes)}

    # Sample user distribution for each class using Dirichlet distribution
    for c in range(num_classes):
        # Draw samples from a Dirichlet distribution
        dirichlet_dist = np.random.dirichlet(np.ones(num_users) * alpha, size=1)[0]

        # Get indices for the current class
        idxs = classwise_idxs[c]

        # Calculate number of samples per user for this class
        num_samples_per_user = np.round(dirichlet_dist * len(idxs)).astype(int)

        # Shuffle indices to ensure random distribution
        np.random.shuffle(idxs)

        # Assign samples to each user based on the sampled distribution
        start = 0
        for user, num_samples in enumerate(num_samples_per_user):
            end = start + num_samples
            # Ensure concatenation doesn't change dtype and indices are kept as integers
            dict_users[user] = np.concatenate((dict_users[user], idxs[start:end].astype(np.int64)), axis=0)
            start = end

    # Ensure all indices are integers
    for user in dict_users.keys():
        dict_users[user] = np.array(dict_users[user], dtype=np.int64)

    return dict_users

def mnist_iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):

    num_shards, num_imgs = 200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar10_noniid(dataset, num_users):

    # Assume CIFAR-10 like dataset with 10 classes
    num_shards, num_imgs = 200, 50  # Adjust based on CIFAR-10 dataset size
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # Each user gets data from 2 shards
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
