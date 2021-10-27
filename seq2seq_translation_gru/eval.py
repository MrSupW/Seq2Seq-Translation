# -*-coding:utf-8-*-
import jieba
import torch

from data import SOS_TOKEN, EOS_TOKEN, src_word2index, trg_index2word
from settings import MAX_LENGTH, device

encoder = torch.load('models/best_encoder.pth').to(device)
decoder = torch.load('models/best_decoder.pth').to(device)


def process_sentence(sentence):
    words = jieba.lcut(sentence)
    words_vec = [src_word2index[word] for word in words]
    return words_vec


def translate_sentence(sentence):
    _, hidden = encoder(torch.LongTensor(process_sentence(sentence)).unsqueeze(0).to(device))
    outputs = []
    output = torch.LongTensor([SOS_TOKEN]).to(device)
    for i in range(MAX_LENGTH):
        output, hidden = decoder(output, hidden)
        output = output.argmax(1)
        out_index = output[0].item()
        if out_index == EOS_TOKEN:
            break
        outputs.append(out_index)
    return ' '.join(trg_index2word[ind] for ind in outputs)


while True:
    src_sentence = input('please input the chinese:')
    if src_sentence == 'quit':
        break
    trans_res = translate_sentence(src_sentence)
    print(f'translation result:{trans_res}')
