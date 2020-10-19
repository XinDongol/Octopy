import torch
import torchvision
import numpy as np


class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.targets = [self.dataset.targets[i] for i in self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, torch.tensor(label)
    
    
def uniform_random_split(dataset, num_clients):
    # https://numpy.org/doc/stable/reference/generated/numpy.array_split.html#numpy.array_split
    split_result = np.array_split(np.arange(len(dataset)).astype(np.int32), num_clients)
    print('Len of last client: %d'%len(split_result[-1]))
    return {idx:i.tolist() for idx, i in enumerate(split_result)}

def non_iid_split(dataset, num_clients, shards_per_client=2):

    num_shards = num_clients * shards_per_client
    num_imgs = len(dataset)//num_shards

    idx_shard = list(range(num_shards))
    dict_users = {i: np.array([], dtype='int32') for i in range(num_clients)}
    idxs = np.arange(len(dataset)).astype(np.int32)
    labels = np.array(dataset.targets).astype(np.int32)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':
    trainset = torchvision.datasets.CIFAR10(root='/cifar', train=True, download=True)
    dict_users = non_iid_split(trainset, 20, 2)
    for l in dict_users.values():
        label = [trainset[i][1] for i in l]
        print(set(label), len(label))

    