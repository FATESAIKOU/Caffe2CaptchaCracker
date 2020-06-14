#!/usr/bin/env python
"""
A program to generate captcha training data

@author:    EricChiang
@date:      2017/07/04
@args
    - digit num
    - sample size for each number
    - db name
    - db type
    - db path
"""

import random
import numpy as np

import sys, os

from PIL import Image
from captcha.image import ImageCaptcha

from caffe2.python import core, utils, workspace
from caffe2.proto import caffe2_pb2


""" Utils Defination"""

def GenProgressBar(now, total, bar_len):
    return '[' + ('=' * int(now * bar_len / total) + '>').ljust(bar_len) + ']'


""" Params Assignment """

DIGITS = int(sys.argv[1])
DATASET_SAMPLE_SIZE = int(sys.argv[2])
DATABASE_NAME = sys.argv[3] + '.' + sys.argv[4]
DATABASE_TYPE = sys.argv[4]
DATABASE_PATH = sys.argv[5]


""" DB Initialization """

image = ImageCaptcha()
db = core.C.create_db(
    DATABASE_TYPE,
    os.path.join(DATABASE_PATH, DATABASE_NAME),
    core.C.Mode.write
)
transaction = db.new_transaction()


""" Meta-info Initialization """

img_data = []
img_label = []
dataset_class_num = 10 ** DIGITS

total = DATASET_SAMPLE_SIZE * dataset_class_num
count = 0
data = range(dataset_class_num)


""" Image Generation """

for _ in xrange(DATASET_SAMPLE_SIZE):
    random.shuffle(data)
    
    for num in data:
        count = count + 1
        if count % 100 == 0:
            os.system('clear')
            print GenProgressBar(count, total, 50), '%d/%d picture(s)' % (count, total)

        text = str(num).zfill(DIGITS)
        captcha = image.generate(text)
        captcha_image = Image.open(captcha).convert('L')

        img_data = np.array(captcha_image.getdata())
        img_label = np.array([int(c) for c in text])
        
        data_and_label = caffe2_pb2.TensorProtos()
        data_and_label.protos.extend([
            utils.NumpyArrayToCaffe2Tensor(img_data)
        ] + [
            utils.NumpyArrayToCaffe2Tensor(img_label[i])
            for i in range(DIGITS)
        ])
        transaction.put(
            'train_%d' % count,
            data_and_label.SerializeToString()
        )

del transaction
del db


""" Report """
print 'Generated db:[' + DATABASE_NAME + '] at ' + DATABASE_PATH + '.'
