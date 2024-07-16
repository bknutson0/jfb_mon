from src.utils.data import load_data

# Test load_data function
#train_loader, test_loader = load_data('synthetic')
train_loader, test_loader = load_data('mnist')

# Print first batch of data
for X, Y in train_loader:
    print(f'{X.shape = }')
    print(f'{Y.shape = }')
    print(Y)
    break