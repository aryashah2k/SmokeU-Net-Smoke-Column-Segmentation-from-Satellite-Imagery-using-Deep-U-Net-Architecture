import torch
from torch.cuda import amp
from torchvision import transforms
from torch.utils.data import DataLoader

from src.data.dataset import SmokeDataset
from src.models.unet import Unet
from src.utils.metrics import metrics
from config.config import Config

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = amp.GradScaler()

    transform_data = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])

    # Create datasets
    train_dataset = SmokeDataset(Config.TRAIN_PATH, Config.TRAIN_MASKS_PATH, 
                                transform_data, transform_data)
    val_dataset = SmokeDataset(Config.VALID_PATH, Config.VALID_MASKS_PATH, 
                              transform_data, transform_data)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Initialize model
    model = Unet(3, 64, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, 
                               momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-1,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.43,
        div_factor=10,
        final_div_factor=1000,
        three_phase=True
    )

    # Training loop implementation here...

if __name__ == "__main__":
    main()