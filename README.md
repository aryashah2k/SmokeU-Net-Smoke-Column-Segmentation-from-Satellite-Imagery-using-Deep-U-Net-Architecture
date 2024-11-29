# SmokeU-Net-Smoke-Column-Segmentation-from-Satellite-Imagery-using-Deep-U-Net-Architecture

--------------

This research proposes an automated deep learning approach for segmenting smoke columns in satellite imagery using U-Net architecture. Current manual and semi-automated methods for smoke plume detection are time-consuming and prone to human error. We aim to develop a robust segmentation model trained on multi-spectral satellite data that can accurately identify and segment smoke columns from wildfires. The proposed framework will leverage U-Net's encoder-decoder architecture to process multiple spectral bands and generate precise smoke column masks. Through this work, we expect to demonstrate significant improvements in smoke plume detection accuracy and processing efficiency compared to traditional threshold-based methods. This research will contribute to advancing automated wildfire monitoring capabilities and support early warning systems for disaster management. 

The model processes multi-spectral satellite data to generate precise smoke column masks, improving upon manual detection methods. Results demonstrate high accuracy in smoke plume detection with 99.04% validation accuracy and 0.91 Dice coefficient.

----------------

## Project Structure

Follow this structure for replication or simply run the Notebook in the Notebooks Folder


```
smoke_segmentation/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── dataloader.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py
│   │   └── layers.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── training/
│       ├── __init__.py
│       └── trainer.py
├── config/
│   └── config.py
├── train.py
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aryashah2k/SmokeU-Net-Smoke-Column-Segmentation-from-Satellite-Imagery-using-Deep-U-Net-Architecture.git
cd smoke-segmentation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib pillow opencv-python pandas
```

## Usage

### Training

1. Configure your dataset paths and training parameters in `config/config.py`

2. Run training:
```bash
python train.py
```

### Inference

```python
import torch
from src.models.unet import Unet
from torchvision import transforms
from PIL import Image

# Load model
model = Unet(channels_in=3, channels=64, num_classes=2)
model.load_state_dict(torch.load('path/to/model.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor()
])

# Load and process image
image = Image.open('path/to/image.jpg')
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
```

## Dataset Format

The dataset should be organized as follows:
```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── train_masks/
│   ├── image1_mask.jpg
│   ├── image2_mask.jpg
│   └── ...
├── valid/
├── valid_masks/
├── test/
└── test_masks/
```

## Model Architecture

The implementation uses a U-Net architecture with:
- Input channels: 3 (RGB)
- Initial features: 64
- Output classes: 2 (binary segmentation)
- Optimization: SGD with OneCycleLR scheduler
- Loss function: Cross-entropy

## Training Parameters

Default training configuration:
- Batch size: 32
- Learning rate: 0.01
- Momentum: 0.95
- Weight decay: 1e-4
- Epochs: 25
- Image size: 128x128

## Performance Metrics

The model is evaluated using:
- Accuracy
- Dice coefficient
- IoU (Intersection over Union)
- Confusion matrix metrics (TP, TN, FP, FN)






