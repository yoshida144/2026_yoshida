from torchvision import datasets
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset"

def load_config(file_path):
    print(f"Using config-path: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def load_dataset_info(dataset_name):
    with open(DATA_DIR / "datasets_info.json", 'r', encoding='utf-8') as f:
        info = json.load(f)
    return info[dataset_name]

def load_dataset(dataset_name, train_transform, test_transform):
    print(f"Loading dataset: {dataset_name}")
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=test_transform)
    elif dataset_name == "KMNIST":
        train_dataset = datasets.KMNIST(root=DATA_DIR, train=True, download=True, transform=train_transform)
        test_dataset = datasets.KMNIST(root=DATA_DIR, train=False, download=True, transform=test_transform)
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=train_transform)
        test_dataset = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=test_transform)
    elif dataset_name == "SVHN":
        train_dataset = datasets.SVHN(root=DATA_DIR, train=True, download=True, transform=train_transform)
        test_dataset = datasets.SVHN(root=DATA_DIR, train=False, download=True, transform=test_transform)
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
    
    return train_dataset, test_dataset
