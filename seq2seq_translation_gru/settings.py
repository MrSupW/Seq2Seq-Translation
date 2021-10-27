# -*-coding:utf-8-*-
import torch
from logger import *

MAX_LENGTH = 30

BATCH_SIZE = 160
EMBED_DIM = 512
HIDDEN_DIM = 1024
N_LAYERS = 6
DROPOUT = 0.5
LR = 0.0001
GAMMA = 0.8
CLIP = 1

N_EPOCHS = 100
TEACHER_FORCING = 0.5
SRC_VOCAB_MAX_SIZE = 150000
TRG_VOCAB_MAX_SIZE = 100000

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

log_info(f'device is {device}')
