# -*-coding:utf-8-*-
import os.path
import time
import torch.nn.utils
from matplotlib import pyplot as plt

from data import *
from model import Transformer


def train(dataloader):
    global train_iter
    transformer.train()
    train_loss = 0.0
    train_count = 0
    for src_batch, trg_batch in dataloader:
        optimizer.zero_grad()
        output = transformer(src_batch, trg_batch[:-1, :])
        output = output.reshape(-1, output.shape[2])
        trg_batch = trg_batch[1:].reshape(-1)
        loss = loss_func(output, trg_batch)
        train_loss += loss.item()
        iter_losses[train_iter % log_per_iter] = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), CLIP)
        optimizer.step()
        if not (train_iter + 1) % log_per_iter:
            avg_loss = iter_losses.mean().item()
            train_losses.append(avg_loss)
            log_info(f'\ttrain Iter {train_iter + 1} Loss {avg_loss:.4f}')
        train_iter += 1
        train_count += 1
    return train_loss / train_count


def evaluate(dataloader):
    transformer.eval()
    eval_loss = 0.0
    eval_count = 0
    with torch.no_grad():
        for src_batch, trg_batch in dataloader:
            output = transformer(src_batch, trg_batch[:-1, :])
            output = output.reshape(-1, output.shape[2])
            trg_batch = trg_batch[1:].reshape(-1)
            loss = loss_func(output, trg_batch)
            eval_loss += loss.item()
            eval_count += 1
    return eval_loss / eval_count


def draw_plot():
    plt.plot(range(log_per_iter, len(train_losses) * log_per_iter + 1, log_per_iter), train_losses)
    plt.title("Train Loss Chart")
    plt.xlabel(f'train iter (batch size = {BATCH_SIZE})')
    plt.ylabel('loss')
    plt.savefig('charts/train_loss.png', dpi=1200)
    plt.clf()

    plt.plot(range(1, len(epoch_train_losses) + 1), epoch_train_losses)
    plt.title("Epoch Train Loss Chart")
    plt.xlabel(f'train epoch (batch size = {BATCH_SIZE})')
    plt.ylabel('loss')
    plt.savefig('charts/epoch_train_loss.png', dpi=1200)
    plt.clf()

    plt.plot(range(1, len(epoch_valid_losses) + 1), epoch_valid_losses)
    plt.title("Epoch Valid Loss Chart")
    plt.xlabel(f'train epoch (batch size = {BATCH_SIZE})')
    plt.ylabel('loss')
    plt.savefig('charts/epoch_valid_loss.png', dpi=1200)
    plt.clf()


if __name__ == '__main__':
    log_info('Train Start'.center(50, '-'))
    log_info(f"""Settings
    MAX_LENGTH = {MAX_LENGTH}
    
    N_EPOCHS = {N_EPOCHS}
    BATCH_SIZE = {BATCH_SIZE}
    EMBED_DIM = {EMBED_DIM}
    NUM_HEADS = {NUM_HEADS}
    NUM_ENCODER_LAYERS = {NUM_ENCODER_LAYERS}
    NUM_DECODER_LAYERS = {NUM_DECODER_LAYERS}
    FORWARD_EXPANSION = {FORWARD_EXPANSION}
    DROPOUT = {DROPOUT}
    LR = {LR}
    GAMMA = {GAMMA}
    CLIP = {CLIP}
    
    SRC_VOCAB_MAX_SIZE = {SRC_VOCAB_MAX_SIZE}
    TRG_VOCAB_MAX_SIZE = {TRG_VOCAB_MAX_SIZE}
    USE_MULTI_GPU = {USE_MULTI_GPU}
    DEVICE = {DEVICE}
    DEVICES = {DEVICES}\n""")

    transformer = Transformer(
        embed_dim=EMBED_DIM,
        src_vocab_size=src_n_words,
        trg_vocab_size=trg_n_words,
        src_pad_index=PAD_TOKEN,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        forward_expansion=FORWARD_EXPANSION,
        dropout=DROPOUT,
        max_len=MAX_LENGTH + 2,
        device=DEVICE
    ).to(DEVICE)

    if USE_MULTI_GPU and DEVICES:
        transformer = torch.nn.DataParallel(transformer, device_ids=DEVICES, dim=1)

    log_info("Model")
    log_info(transformer)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=LR)

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN).to(DEVICE)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=GAMMA)

    train_iter = 0
    log_per_iter = 750
    iter_losses = torch.zeros(log_per_iter).to(DEVICE)
    best_valid_loss = float('inf')

    train_losses = []
    epoch_train_losses = []
    epoch_valid_losses = []
    if not os.path.isdir("models"):
        os.mkdir("models")

    for i in range(1, N_EPOCHS + 1):
        start_time = time.time()
        log_info(f'Epoch: {i}')
        train_loss = train(train_data_loader)
        valid_loss = evaluate(valid_data_loader)
        log_info(f"""Epoch: {i}  | time in {(time.time() - start_time) // 60:.0f}m {(time.time() - start_time) % 60:.1f}s
        Loss: {train_loss:>6.4f}(train) |   Acc: {0 * 100:.2f}%(train)
        Loss: {valid_loss:>6.4f}(valid) |   Acc: {0 * 100:.2f}%(valid)\n""")
        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)
        draw_plot()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            if USE_MULTI_GPU and DEVICES:
                torch.save(transformer.module.state_dict(), "models/transformer-state-best.pth")
            else:
                torch.save(transformer.state_dict(), 'models/transformer-state-best.pth')
        else:
            scheduler.step()
    log_info('Train Finish'.center(50, '-'))
