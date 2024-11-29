import os

class Config:
    # Paths
    BASE_PATH = os.path.join(os.path.dirname(os.getcwd()), 
                            '/kaggle/input/goes16-wildfires-smoke-plumes-dataset/GOES16-wildfires-smoke-plumes-dataset-1da')
    TRAIN_PATH = os.path.join(BASE_PATH, 'train')
    TRAIN_MASKS_PATH = os.path.join(BASE_PATH, 'train_masks')
    VALID_PATH = os.path.join(BASE_PATH, 'valid')
    VALID_MASKS_PATH = os.path.join(BASE_PATH, 'valid_masks')
    TEST_PATH = os.path.join(BASE_PATH, 'test')
    TEST_MASKS_PATH = os.path.join(BASE_PATH, 'test_masks')
    MODELS_PATH = "/kaggle/working/"

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 0.01
    MOMENTUM = 0.95
    WEIGHT_DECAY = 1e-4