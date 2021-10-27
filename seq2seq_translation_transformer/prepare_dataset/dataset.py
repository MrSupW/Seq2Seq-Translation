# -*-coding:utf-8-*-
import re

import jieba
import unicodedata

from seq2seq_translation_transformer.prepare_dataset.zhtool.langconv import Converter


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


process_zh_sentence = lambda x: ' '.join(jieba.lcut(Converter('zh-hans').convert(x).encode('utf-8')))
process_en_sentence = lambda x: en_normalizeString(x)

# # TODO Process chs-eng-cut.txt
# print("Processing chs-eng-cut.txt...")
# lines = open('../data/origin/chs-eng.txt', encoding='UTF-8').read().strip().split('\n')
# count = 0
# print('Preparing dataset...')
# with open('../data/chs-eng.txt', 'w', encoding='utf-8') as f:
#     for line in lines:
#         try:
#             zh, en = line.split("\t")
#             if not en or not zh:
#                 continue
#             f.write(process_zh_sentence(zh) + '\t' + process_en_sentence(en) + '\n')
#             count += 1
#             if not count % 10000:
#                 print(f'read {count} lines')
#         except Exception:
#             continue

# TODO  Process translation2019zh_valid.json
# import json
#
# count = 0
# lines = open('../data/origin/translation2019zh_train.json', encoding='utf-8').read().strip().split('\n')
# with open('../data/translation2019zh-train-cut-small.txt', 'w', encoding='utf-8') as f:
#     for line in lines:
#         item = json.loads(line)
#         f.write(process_zh_sentence(item['chinese']) + '\t' + process_en_sentence(item['english']) + '\n')
#         count += 1
#         if not count % 10000:
#             print(f'read {count} lines')


# # TODO Process news-commentary-v14.en-zh.tsv
# print('Process news-commentary-v14.en-zh.tsv...')
# count = 0
# lines = open('../data/origin/news-commentary-v14.en-zh.tsv', encoding='utf-8').read().strip().split('\n')
# with open('../data/news-commentary-v14.en-zh.txt', 'w', encoding='utf-8') as f:
#     for line in lines:
#         try:
#             en, zh = line.split("\t")
#             if not en or not zh:
#                 continue
#             f.write(process_zh_sentence(zh) + '\t' + process_en_sentence(en) + '\n')
#             count += 1
#             if not count % 10000:
#                 print(f'read {count} lines')
#         except Exception:
#             continue
#
# # TODO Process wikititles-v1.zh-en.tsv
# print("Process wikititles-v1.zh-en.tsv...")
# count = 0
# lines = open('../data/origin/wikititles-v1.zh-en.tsv', encoding='utf-8').read().strip().split('\n')
# with open('../data/wikititles-v1.zh-en.txt', 'w', encoding='utf-8') as f:
#     for line in lines:
#         try:
#             zh, en = line.split("\t")
#             if not en or not zh:
#                 continue
#             f.write(process_zh_sentence(zh) + '\t' + process_en_sentence(en) + '\n')
#             count += 1
#             if not count % 10000:
#                 print(f'read {count} lines')
#         except Exception:
#             continue

# TODO Merge all the dataset
# print("Merge all the dataset")
# f1 = open("../data/train.txt", encoding='utf-8').readlines()[:-1]
# f2 = open("../data/chs-eng.txt", encoding='utf-8').readlines()[:-1]
# f3 = open("../data/news-commentary-v14.en-zh.txt", encoding='utf-8').readlines()[:-1]
#
# with open("../data/train.txt", 'w', encoding='utf-8') as f:
#     f.writelines(f1)
#     f.writelines(f2)
#     f.writelines(f3)
