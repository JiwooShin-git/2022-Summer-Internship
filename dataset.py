from torchvision import transforms, datasets
import torch.utils.data as data
import torch
import os
import numpy as np


def create_loader(batch_size, data_dir, data):
    if data.lower() == 'cifar100':
        n_class, n_sample_class = 100, 500
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

        trainset = datasets.CIFAR100(root=os.path.join(data_dir, data), train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=os.path.join(data_dir, data), train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)

    elif data.lower() == 'tiny':
        n_class, n_sample_class = 200, 500

        transform_train = transforms.Compose(
            [transforms.RandomCrop(64, padding=8), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)), ])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)), ])
        trainset = datasets.ImageFolder(root=os.path.join(data_dir, data, 'train'), transform=transform_train)
        testset = datasets.ImageFolder(root=os.path.join(data_dir, data, 'valid'), transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)

    elif data.lower() == 'imagenet':
        n_class, n_sample_class = 1000, 1300

        transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])
        transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])

        trainset = datasets.ImageFolder(root=os.path.join(data_dir, data, 'train'), transform=transform_train)
        testset = datasets.ImageFolder(root=os.path.join(data_dir, data, 'valid'), transform=transform_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                                  num_workers=2)

    return train_loader, test_loader, n_class, n_sample_class

if __name__ == '__main__':
    train_loader, test_loader, _, _ = create_loader(64, r'D:\Image\Image classification', 'imagenet')

