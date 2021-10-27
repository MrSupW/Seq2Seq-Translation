# -*-coding:utf-8-*-
import torch

from logger import *

UNK_TOKEN = 0
PAD_TOKEN = 1
SOS_TOKEN = 2
EOS_TOKEN = 3

N_EPOCHS = 100

MAX_LENGTH = 100
BATCH_SIZE = 56
EMBED_DIM = 768
NUM_HEADS = 12
NUM_ENCODER_LAYERS = 8
NUM_DECODER_LAYERS = 8
FORWARD_EXPANSION = 2048
DROPOUT = 0.10
LR = 0.00005
GAMMA = 0.5
CLIP = 1

SRC_VOCAB_MAX_SIZE = 300000
TRG_VOCAB_MAX_SIZE = 150000

USE_MULTI_GPU = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
DEVICES = [] if not torch.cuda.is_available() else list(range(torch.cuda.device_count()))

log_info(f'device is {DEVICE}')
log_info(f'all devices id are {DEVICES}')
