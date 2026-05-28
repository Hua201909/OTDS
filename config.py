import torch

# Path Configuration
DATA_DIR = "./data"
TRAIN_PATH = f"{DATA_DIR}/train.xlsx"
TEST_PATH  = f"{DATA_DIR}/test.xlsx"

# Model Hyperparameters
SEQ_LENGTH = 128
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 20
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-5

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 1234