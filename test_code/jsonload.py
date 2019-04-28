# -*- coding:utf8 -*-
# @TIME     : 2019/4/17 19:47
# @Author   : SuHao
# @File     : json.py


import json


file = 'H:\\毕业设计\\普通话数据集1\\primewords_md_2018_set1\\set1_transcript.json'
fb = open(file, 'r')
dicts = json.loads(fb)
fb.close()
