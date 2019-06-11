from torchvision import datasets, transforms
import os
def load_visda():
    train_path = '../visda_classification/visda_datasets/train'
    val_path = '../visda_classification/visda_datasets/validation'
    data_transforms = {
        train_path: transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        val_path: transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [train_path, val_path]}
    dset_sizes = {x: len(dsets[x]) for x in [train_path, val_path]}
    dset_classes = dsets[train_path].class_to_idx
    print('classes' + str(dset_classes))
    S = dsets[train_path]
    T = dsets[val_path]
    return S, T