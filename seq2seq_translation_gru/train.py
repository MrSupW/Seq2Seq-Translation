# -*-coding:utf-8-*-
import os
import random
import time

import matplotlib.pyplot as plt

from data import *
from model import EncoderRNN, DecoderRNN


def train(dataloader):
    global train_iter
    encoder.train()
    decoder.train()
    train_loss = 0
    train_count = 0
    for src_batch, trg_batch in dataloader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss = 0
        batch_size = src_batch.shape[0]
        _, hidden = encoder(src_batch)

        trg_len = trg_batch.shape[1]
        teacher_forcing = random.random() < TEACHER_FORCING
        output = torch.LongTensor([SOS_TOKEN for _ in range(batch_size)]).to(device)
        if teacher_forcing:
            for ind in range(trg_len):
                output, hidden = decoder(output, hidden)
                loss += loss_func(output, trg_batch[:, ind])
                output = trg_batch[:, ind]
        else:
            for ind in range(trg_len):
                output, hidden = decoder(output, hidden)
                loss += loss_func(output, trg_batch[:, ind])
                output = output.argmax(1)
        train_loss += loss.item()
        iter_losses[train_iter % log_per_iter] = loss.item()

        if not (train_iter + 1) % log_per_iter:
            avg_loss = iter_losses.mean().item()
            train_losses.append(avg_loss)
            log_info(f'\ttrain Iter {train_iter + 1} Loss {avg_loss:.4f}')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), CLIP)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)
        encoder_optimizer.step()
        decoder_optimizer.step()
        train_iter += 1
        train_count += 1
    return train_loss / train_count


def evaluate(dataloader):
    encoder.eval()
    decoder.eval()

    eval_loss = 0
    eval_count = 0
    with torch.no_grad():
        for src_batch, trg_batch in dataloader:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0
            batch_size = src_batch.shape[0]
            _, hidden = encoder(src_batch)

            trg_len = trg_batch.shape[1]
            output = torch.LongTensor([SOS_TOKEN for _ in range(batch_size)]).to(device)
            for ind in range(trg_len):
                output, hidden = decoder(output, hidden)
                loss += loss_func(output, trg_batch[:, ind])
                output = output.argmax(1)
            eval_loss += loss.item()

            encoder_optimizer.step()
            decoder_optimizer.step()
            eval_count += 1
    return eval_loss / eval_count


if __name__ == '__main__':
    log_info('Train Start'.center(50, '-'))
    log_info(f"""Settings:
    MAX_LENGTH = {MAX_LENGTH}
    BATCH_SIZE = {BATCH_SIZE}
    EMBED_DIM = {EMBED_DIM}
    HIDDEN_DIM = {HIDDEN_DIM}
    N_LAYERS = {N_LAYERS}
    DROPOUT = {DROPOUT}
    LR = {LR}
    GAMMA = {GAMMA}
    CLIP = {CLIP}
    N_EPOCHS = {N_EPOCHS}
    TEACHER_FORCING = {TEACHER_FORCING}\n""")

    encoder = EncoderRNN(src_n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)
    decoder = DecoderRNN(trg_n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    loss_func = torch.nn.CrossEntropyLoss()

    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, 1, gamma=GAMMA)
    decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, 1, gamma=GAMMA)

    train_iter = 0
    log_per_iter = 500
    iter_losses = torch.zeros(log_per_iter).to(device)
    best_valid_loss = float('inf')

    train_losses = []

    if not os.path.isdir("models"):
        os.mkdir("models")

    for i in range(1, N_EPOCHS + 1):
        start_time = time.time()
        log_info(f'Epoch: {i}')
        train_loss = train(train_data_loader)
        valid_loss = evaluate(valid_data_loader)
        log_info(f"""Epoch: {i}  | time in {int((time.time() - start_time) // 60)}m {(time.time() - start_time) % 60:.1f}s
        Loss: {train_loss:>6.4f}(train) |   Acc: {0 * 100:.2f}%(train)
        Loss: {valid_loss:>6.4f}(valid) |   Acc: {0 * 100:.2f}%(valid)\n""")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(encoder, 'models/best_encoder_big.pth')
            torch.save(decoder, 'models/best_decoder_big.pth')
        else:
            encoder_scheduler.step()
            decoder_scheduler.step()
    log_info('Train Finish'.center(50, '-'))
    # 进行绘图
    plt.plot(range(log_per_iter, len(train_losses) * log_per_iter + 1, log_per_iter), train_losses)
    plt.title("Train Loss Chart")
    plt.xlabel(f'train iter (batch size = {BATCH_SIZE})')
    plt.ylabel('loss')
    plt.savefig('train_loss.png', dpi=1200)
    plt.clf()