# -*-coding:utf-8-*-
import re

import jieba
import unicodedata


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def en_normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([,.!?;])", r" \1", s)
    s = re.sub(r"([\":-])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?'\";:\d-]+", r" ", s)
    return s


process_zh_sentence = lambda x: ' '.join(jieba.lcut(x))
process_en_sentence = lambda x: en_normalizeString(x)

# Process chs-eng-cut.txt
# lines = open('data/origin/chs-eng-cut.txt', encoding='UTF-8').read().strip().split('\n')
#
# lines = [(line.split('\t')[0], line.split('\t')[1]) for line in lines]
# print('Preparing dataset...')
# with open('data/chs-eng-cut.txt', 'w', encoding='UTF-8') as f:
#     for src, trg in lines:
#         f.write(process_zh_sentence(src) + '\t' + process_en_sentence(trg) + '\n')
#
# print('Dataset ready')


# Process translation2019zh_valid.json
import json

count = 0
lines = open('data/origin/translation2019zh_train.json', encoding='utf-8').read().strip().split('\n')[:1000000]
with open('data/translation2019zh-train-cut-small.txt', 'w', encoding='utf-8') as f:
    for line in lines:
        item = json.loads(line)
        f.write(process_zh_sentence(item['chinese']) + '\t' + process_en_sentence(item['english']) + '\n')
        count += 1
        if not count % 10000:
            print(f'read {count} lines')
