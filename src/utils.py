from torchvision import datasets
import json

def load_config(file_path):
    print(f"Using config-path: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def load_dataset_info(dataset_name):
    with open("./datasets/datasets_info.json", 'r', encoding='utf-8') as f:
        info = json.load(f)
    return info[dataset_name]

def load_dataset(dataset_name, train_transform, test_transform):
    print(f"Loading dataset: {dataset_name}")
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root="./datasets", train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root="./datasets", train=False, download=True, transform=test_transform)
    elif dataset_name == "KMNIST":
        train_dataset = datasets.KMNIST(root="./datasets", train=True, download=True, transform=train_transform)
        test_dataset = datasets.KMNIST(root="./datasets", train=False, download=True, transform=test_transform)
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root="./datasets", train=True, download=True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(root="./datasets", train=False, download=True, transform=test_transform)
    elif dataset_name == "SVHN":
        train_dataset = datasets.SVHN(root="./datasets", train=True, download=True, transform=train_transform)
        test_dataset = datasets.SVHN(root="./datasets", train=False, download=True, transform=test_transform)
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="./datasets", train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root="./datasets", train=False, download=True, transform=test_transform)
    
    return train_dataset, test_dataset
