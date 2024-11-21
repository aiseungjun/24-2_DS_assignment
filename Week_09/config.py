import torch

BATCH_SIZE = 2
NUM_WORKERS = 4
EPOCHS = 10
LEARNING_RATE = 0.001

DATA_DIR = './data/UCF-101'
ANNOTATION_PATH = './data/ucfTrainTestlist'

TRAIN_SPLIT = 0.8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
