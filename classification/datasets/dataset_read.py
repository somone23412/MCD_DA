import sys

sys.path.append('../loader')
from datasets.unaligned_data_loader import UnalignedDataLoader
from datasets.svhn import load_svhn
from datasets.mnist import load_mnist
from datasets.usps import load_usps
from datasets.gtsrb import load_gtsrb
from datasets.synth_traffic import load_syntraffic
from datasets.visda import load_visda
from datasets.office31 import load_office
import torchvision.transforms as transforms
from datasets.dataset import Dataset

def return_dataset(data, scale=False, usps=False, all_use='no'):
    if data == 'svhn':
        train_image, train_label, \
        test_image, test_label = load_svhn()
    if data == 'mnist':
        train_image, train_label, \
        test_image, test_label = load_mnist(scale=scale, usps=usps, all_use=all_use)
    if data == 'usps':
        train_image, train_label, \
        test_image, test_label = load_usps(all_use=all_use)
    if data == 'synth':
        train_image, train_label, \
        test_image, test_label = load_syntraffic()
    if data == 'gtsrb':
        train_image, train_label, \
        test_image, test_label = load_gtsrb()
    print(data, "\t[train]\t", train_image.shape, "\t[test]\t", test_image.shape)
    return train_image, train_label, test_image, test_label


def trans(source, target, scale=40):
    transform = transforms.Compose([
        ## transforms.Scale(scale),
        transforms.Resize(scale),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_source = Dataset(source['imgs'], source['labels'], transform=transform)
    dataset_target = Dataset(target['imgs'], target['labels'], transform=transform)
    return dataset_source, dataset_target

def dataset_read(source, target, batch_size, scale=False, all_use='no'):
    S = {}
    S_test = {}
    T = {}
    T_test = {}
    if source == 'visda' or source == 'office':
        if source == 'visda':
            S, T = load_visda()
        if source == 'office':
            S, T = load_office()
        S_test = S
        T_test = T
    else:
        usps = False
        if source == 'usps' or target == 'usps':
            usps = True

        train_source, s_label_train, test_source, s_label_test = return_dataset(source, scale=scale,
                                                                                usps=usps, all_use=all_use)
        train_target, t_label_train, test_target, t_label_test = return_dataset(target, scale=scale, usps=usps,
                                                                                all_use=all_use)


        S['imgs'] = train_source
        S['labels'] = s_label_train
        T['imgs'] = train_target
        T['labels'] = t_label_train

        S_test['imgs'] = test_source
        S_test['labels'] = s_label_test
        T_test['imgs'] = test_target
        T_test['labels'] = t_label_test

        ## Trans, scale = (40 if source == 'synth') else (28 if source == 'usps' or target == 'usps') else 32
        scale = 40 if source == 'synth' else 28 if source == 'usps' or target == 'usps' else 32
        S, T = trans(S, T, scale=scale)
        S_test, T_test = trans(S_test, T_test, scale=scale)
    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, scale=scale)
    dataset = train_loader.load_data()

    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size, scale=scale)
    dataset_test = test_loader.load_data()

    return dataset, dataset_test
