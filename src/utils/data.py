import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.utils.config import default_config

# Get training configuration
dataset_size = default_config['dataset_size']
train_fraction = default_config['train_fraction']
train_size = round(train_fraction * dataset_size)
test_size = dataset_size - train_size
batch_size = default_config['batch_size']

def load_data(dataset, True_Model=None):
    """ Load data for training and testing. """

    if dataset == 'synthetic' and True_Model is not None:
        # Initialize the true model
        Model = True_Model['class']
        new_config = True_Model['new_config']
        config = {**default_config, **new_config} 
        model = Model(config)
        model.eval() # No need to train

        # Generate random data
        X = torch.randn(dataset_size, config['in_dim'])
        Y = model(X)

        # Save the data
        dataset_dict = {'X': X, 'Y': Y}
        torch.save(dataset_dict, 'data/synthetic.pth')

        # Split the data into training and testing
        dataset = torch.utils.data.TensorDataset(X, Y)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset == 'mnist':
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load MNIST dataset
        train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

        # Create data loaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        # Verify
        for images, labels in train_loader:
            print(images.shape, labels.shape)
            break

    return train_loader, test_loader
    