# -*-coding:utf-8-*-

import os
import datetime


def log_info(info):
    print(info)
    logger.write((datetime.datetime.now() + datetime.timedelta(hours=8))
                 .strftime('[%Y/%m/%d %H:%M:%S]') + ' ' + str(info) + '\n')
    logger.flush()


if not os.path.isdir("logs"):
    os.mkdir("logs")
logger = open(f'logs/log'
              f'{(datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y_%m_%d-%H_%M_%S")}'
              f'.txt', 'w', encoding='UTF-8')
