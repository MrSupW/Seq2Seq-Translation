# -*-coding:utf-8-*-

import jieba

from data import *
from seq2seq_translation_transformer.model import Transformer
from settings import *


def translate(model, sentence):
    sentence = jieba.lcut(sentence)
    sent_vec = src_pipeline(sentence)
    sent_tensor = torch.LongTensor(sent_vec).unsqueeze(1).to(DEVICE)
    outputs = [SOS_TOKEN]
    with torch.no_grad():
        for _ in range(MAX_LENGTH):
            trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(DEVICE)
            # output -> [trg_seq_len, batch, trg_vocab_size]
            output = model(sent_tensor, trg_tensor)
            output = output[-1]
            word_index = output[0].argmax(0).item()
            if word_index == EOS_TOKEN:
                break
            outputs.append(word_index)
    res = ' '.join(trg_index2word.get(token, '<UNK>') for token in outputs[1:]).capitalize()
    res = res.replace(" ,", ',').replace(" .", '.').replace(" ?", "?")
    return res


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
    max_len=MAX_LENGTH + 2,  # 2表示SOS和EOS
    device=DEVICE
).to(DEVICE)

transformer.load_state_dict(torch.load("models/transformer-state.pth"))

while True:
    sent = input("Please input the chinese:")
    if sent == 'quit':
        break
    print(f"translation result: '{translate(transformer, sent)}'")
