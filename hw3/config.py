import torch

# ====== GENERAL ======
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== DATA ======
DATA_ROOT = "./data"
N_PERCENT = 10
BATCH_SIZE = 64
NUM_WORKERS = 2
IMG_SIZE = 96

# ====== TRAINING ======
EPOCHS_SUP = 30
EPOCHS_SSL = 100

LR_SUP = 3e-4
LR_SSL = 1e-3

# ====== MODEL ======
NUM_CLASSES = 10
BACKBONE = "efficientnet_b0"

# ====== PATHS ======
SSL_ENCODER_PATH = "hw3/SSL/ssl_encoder.pth"