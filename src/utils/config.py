default_config = {
    # Device
    'preferred_cuda': 1,

    # Model
    'in_dim': 10,
    'out_dim': 10,
    'max_iter': 100,
    'tol':1e-6,
    'm':1,
    'decay':0.5,

    # Training
    'dataset_size': 1000,
    'train_fraction': 0.8,
    'batch_size': 32,
    'lr': 1,
    'max_epochs': 10,
}