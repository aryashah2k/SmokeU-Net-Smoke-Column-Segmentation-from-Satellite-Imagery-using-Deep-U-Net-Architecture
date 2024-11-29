from torch.utils.data import DataLoader

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Creates DataLoader objects for training, validation, and testing datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Size of each batch (default: 32)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    
    return train_loader, val_loader, test_loader