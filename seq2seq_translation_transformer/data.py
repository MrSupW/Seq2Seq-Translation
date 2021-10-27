# -*-coding:utf-8-*-
import math

from torch.utils.data import random_split, DataLoader

from settings import *

log_info('Data Loading'.center(50, '-'))
lines = []

for line in open("data/train.txt", encoding='UTF-8').read().strip().split('\n'):
    src, trg = line.split('\t')
    lines.append((src.split(), trg.split()))

log_info(f'data size: {len(lines)}')

# 进行MAX_LENGTH过滤
cut_lines = []
for src, trg in lines:
    if len(src) > MAX_LENGTH or len(trg) > MAX_LENGTH:
        continue
    cut_lines.append((src, trg))

log_info(f'after cut for MAX_LENGTH({MAX_LENGTH}) data size is {len(cut_lines)}')

src_word2index = {}
src_index2word = {UNK_TOKEN: '<UNK>', PAD_TOKEN: '<PAD>', SOS_TOKEN: '<SOS>', EOS_TOKEN: '<EOS>'}
src_word2count = {}
src_n_words = 4

trg_word2index = {}
trg_index2word = {UNK_TOKEN: '<UNK>', PAD_TOKEN: '<PAD>', SOS_TOKEN: '<SOS>', EOS_TOKEN: '<EOS>'}
trg_word2count = {}
trg_n_words = 4

for src, trg in cut_lines:
    for word in set(src):
        if word not in src_word2index:
            src_word2index[word] = src_n_words
            src_index2word[src_n_words] = word
            src_word2count[word] = 1
            src_n_words += 1
        else:
            src_word2count[word] += 1
    for word in set(trg):
        if word not in trg_word2index:
            trg_word2index[word] = trg_n_words
            trg_index2word[trg_n_words] = word
            trg_word2count[word] = 1
            trg_n_words += 1
        else:
            trg_word2count[word] += 1

log_info(f'size of src vocab is {src_n_words}')
log_info(f'size of trg vocab is {trg_n_words}')

log_info(f'count of 你好 is {src_word2count["你好"]}')
log_info(f'count of hello is {trg_word2count["hello"]}')

# 压缩词表 将频率较小的词设为 UNK
if src_n_words > SRC_VOCAB_MAX_SIZE or trg_n_words > TRG_VOCAB_MAX_SIZE:
    log_info('vocab compressing')
    src_word_count = [(key, value) for key, value in src_word2count.items()]
    trg_word_count = [(key, value) for key, value in trg_word2count.items()]

    src_word_count.sort(key=lambda x: x[1], reverse=True)
    trg_word_count.sort(key=lambda x: x[1], reverse=True)

    src_words = [w for w, _ in src_word_count[:SRC_VOCAB_MAX_SIZE]]
    trg_words = [w for w, _ in trg_word_count[:TRG_VOCAB_MAX_SIZE]]
    src_index2word = {UNK_TOKEN: '<UNK>', PAD_TOKEN: '<PAD>', SOS_TOKEN: '<SOS>', EOS_TOKEN: '<EOS>'}
    src_word2index = {}
    trg_index2word = {UNK_TOKEN: '<UNK>', PAD_TOKEN: '<PAD>', SOS_TOKEN: '<SOS>', EOS_TOKEN: '<EOS>'}
    trg_word2index = {}

    for i in range(len(src_words)):
        src_index2word[4 + i] = src_words[i]
        src_word2index[src_words[i]] = 4 + i

    for i in range(len(trg_words)):
        trg_index2word[4 + i] = trg_words[i]
        trg_word2index[trg_words[i]] = 4 + i

    src_n_words = len(src_word2index)
    trg_n_words = len(trg_word2index)
    log_info(f'min count of src is {src_word2count[src_words[-1]]}')
    log_info(f'min count of trg is {trg_word2count[trg_words[-1]]}')
    log_info('vocab compress finish')
    src_n_words += 4
    trg_n_words += 4

log_info(f'size of src vocab is {src_n_words}')
log_info(f'size of trg vocab is {trg_n_words}')
log_info(f"hello({trg_word2index['hello']}) world({trg_word2index['world']})")
log_info(f"你好({src_word2index['你好']}) 世界({src_word2index['世界']})")


# 加上特殊标志符


def src_pipeline(words):
    src_vec = [SOS_TOKEN]
    src_vec.extend([src_word2index.get(w, UNK_TOKEN) for w in words])
    src_vec.append(EOS_TOKEN)
    return src_vec


def trg_pipeline(words):
    trg_vec = [SOS_TOKEN]
    trg_vec.extend([trg_word2index.get(w, UNK_TOKEN) for w in words])
    trg_vec.append(EOS_TOKEN)
    return trg_vec


train_data = [(src_pipeline(src), trg_pipeline(trg)) for src, trg in cut_lines]

n_train_data = int(len(train_data) * 0.99)
n_valid_data = len(train_data) - n_train_data
train_data, valid_data = random_split(train_data, [n_train_data, n_valid_data])
train_data = list(train_data)
valid_data = list(valid_data)

log_info(f'train data size: {n_train_data}')
log_info(f'valid data size: {n_valid_data}')

# 按照长度排序
train_data.sort(key=lambda x: len(x[0]) + len(x[1]))
valid_data.sort(key=lambda x: len(x[0]) + len(x[1]))
train_data = [train_data[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(math.ceil(n_train_data / BATCH_SIZE))]
valid_data = [valid_data[i * BATCH_SIZE: (i + 1) * BATCH_SIZE] for i in range(math.ceil(n_valid_data / BATCH_SIZE))]


def generate_batch_data(batch):
    batch = batch[0]
    max_src_len = 0
    max_trg_len = 0
    for src, trg in batch:
        max_src_len = max(max_src_len, len(src))
        max_trg_len = max(max_trg_len, len(trg))
    src_batch = torch.zeros(BATCH_SIZE, max_src_len, dtype=torch.int64).to(DEVICE)
    trg_batch = torch.zeros(BATCH_SIZE, max_trg_len, dtype=torch.int64).to(DEVICE)
    for ind, (src, trg) in enumerate(batch):
        src_batch[ind] = torch.LongTensor(src + [PAD_TOKEN] * (max_src_len - len(src)))
        trg_batch[ind] = torch.LongTensor(trg + [PAD_TOKEN] * (max_trg_len - len(trg)))
    return src_batch.T, trg_batch.T


train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=generate_batch_data)
valid_data_loader = DataLoader(valid_data, batch_size=1, shuffle=False, collate_fn=generate_batch_data)

log_info('Data Ready'.center(50, '-'))
log_info('\n\n\n')
