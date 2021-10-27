# -*-coding:utf-8-*-
import datetime


# import pytz
#
# tz = pytz.timezone('Asia/Shanghai')


def log_info(info):
    print(info)
    logger.write(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' ' + info + '\n')
    logger.flush()


logger = open('log.txt', 'w', encoding='UTF-8')