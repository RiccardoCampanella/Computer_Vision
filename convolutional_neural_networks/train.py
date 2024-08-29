from model import CNNClassifier
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch
import torchvision
import numpy as np
import random
import wandb
import sys
import yaml


def train_collate_fn(batch):
    """Custom collate function for applying transformations to training batches."""
    images, labels = zip(*batch)

    # Apply the transformations directly on tensor images
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25, fill=-1),
    ])
    transformed_images = [transform(image) for image in images]

    # Stack all images into a single batch tensor
    images = torch.stack(transformed_images)
    return images, torch.tensor(labels)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(config, run_name=None):
    set_seed(config['seed'])
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    train_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    train_set, val_set = random_split(train_set, [50000, 10000])

    if 'additional_transforms' in config:
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=train_collate_fn)
    else:
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    if run_name:
        config['run_name'] = run_name
        wandb.init(project='fashion_mnist', config=config, name=run_name)
        wandb.login(key = config['wandb_api_key'])
    else:
        wandb.init(project='fashion_mnist', config=config)
        wandb.login(key = config['wandb_api_key'])
        config['run_name'] = wandb.run.name

    classifier = CNNClassifier(config)
    classifier.fit(train_loader, val_loader)


if __name__ == '__main__':
    # read config yaml file from command line, if command line has second argument then use it as run_name
    config_file = sys.argv[1]
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    if len(sys.argv) == 3:
        run_name = sys.argv[2]
    else:
        run_name = None
    
    main(config, run_name)
