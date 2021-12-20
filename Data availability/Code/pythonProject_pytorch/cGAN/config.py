import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 512
NUM_WORKERS = 4
L1_LAMBDA = 100
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "checkpoints/disc_500.pth.tar"
CHECKPOINT_GEN = "checkpoints/gen_500.pth.tar"

