import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TRAIN_DIR = "../dataset/maps/train"
# VAL_DIR = "../dataset/maps/val"
TRAIN_DIR = '../dataset/Anime/data/train'
VAL_DIR = '../dataset/Anime/data/val'
LEARNING_RATE = 2e-4
BATCH_SIZE = 512
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
# CHECKPOINT_DISC = "checkpoints/Pix2Pix_Satellite_to_Map/disc.pth.tar"
# CHECKPOINT_GEN = "checkpoints/Pix2Pix_Satellite_to_Map/gen.pth.tar"
CHECKPOINT_DISC = "checkpoints/Pix2Pix_Colorize_Anime/disc.pth.tar"
CHECKPOINT_GEN = "checkpoints/Pix2Pix_Colorize_Anime/gen.pth.tar"

