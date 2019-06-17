from torchvision import datasets, transforms
import os
def load_office():
    source_path = 'data/office/domain_adaptation_images/amazon/images'
    target_path = 'data/office/domain_adaptation_images/webcam/images'
    data_transforms = {
        source_path: transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        target_path: transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [source_path, target_path]}
    dset_sizes = {x: len(dsets[x]) for x in [source_path, target_path]}
    dset_classes = dsets[source_path].class_to_idx
    print('classes' + str(dset_classes))
    S = dsets[source_path]
    T = dsets[target_path]
    return S, T