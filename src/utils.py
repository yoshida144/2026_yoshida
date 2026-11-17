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
    from torchvision import datasets
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



# LIF training and evaluation utilities (consolidated here).
import torch

def spike_penalty(model):
    terms = [l.last_spike_loss for l in model.feature_extractor.layers
             if getattr(l, 'last_spike_loss', None) is not None]
    return torch.stack(terms).mean() if terms else next(model.parameters()).new_zeros(())

def set_execution(model, mode):
    if mode not in ('dense', 'event'):
        raise ValueError(mode)
    for layer in model.feature_extractor.layers:
        if hasattr(layer, 'execution_mode'):
            layer.execution_mode = mode
        elif mode == 'event':
            raise ValueError('Event inference requires LIF layers')

def build_loaders(params, classification=False, evaluation_only=False):
    """Original supported torchvision datasets, or offline synthetic smoke data."""
    import json
    from torch.utils.data import DataLoader, TensorDataset
    with open('datasets/datasets_info.json', encoding='utf-8') as f:
        info = json.load(f)[params['target_dataset']]
    if params.get('synthetic', False):
        generator = torch.Generator().manual_seed(43)
        n = int(params.get('synthetic_samples', 8))
        size = info['img_size']
        x = torch.rand(n, info['input_channels'], size, size, generator=generator)
        x = (x-torch.tensor(info['data_mean']).view(1,-1,1,1))/torch.tensor(info['data_std']).view(1,-1,1,1)
        labels = torch.arange(n) % info['num_classes']
        train_set = TensorDataset(x, labels)
        test_set = TensorDataset(x.clone(), labels.clone())
    else:
        from torchvision import transforms, datasets
        name=params['target_dataset']
        aug=[]
        if classification:
            if name=='MNIST':
                aug=[transforms.RandomAffine(10,translate=(0.1,0.1),scale=(0.9,1.1),fill=0)]
            elif name in ('FashionMNIST','CIFAR10'):
                aug=[transforms.RandomCrop(info['img_size'],padding=4,padding_mode='reflect'),transforms.RandomHorizontalFlip()]
        normalize=[transforms.ToTensor(),transforms.Normalize(info['data_mean'],info['data_std'])]
        train_transform=transforms.Compose(aug+normalize)
        test_transform=transforms.Compose(normalize)
        ctor=getattr(datasets,name)
        if name in ('SVHN', 'STL10'):
            train_set=None if evaluation_only else ctor(root='./datasets',split='train',download=True,transform=train_transform)
            test_set=ctor(root='./datasets',split='test',download=True,transform=test_transform)
        else:
            train_set=None if evaluation_only else ctor(root='./datasets',train=True,download=True,transform=train_transform)
            test_set=ctor(root='./datasets',train=False,download=True,transform=test_transform)
    options=dict(batch_size=params['batch_size'],num_workers=0)
    train_loader = None if evaluation_only else DataLoader(train_set,shuffle=True,**options)
    return train_loader,DataLoader(test_set,shuffle=False,**options),info


