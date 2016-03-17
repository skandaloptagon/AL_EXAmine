# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
#
# Author: John Skandalakis

import time
from time import sleep
import gzip
from tldextract import extract
import os

import errno

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

class MessagePackPipeline(object):

    def process_item(self, item, spider):
        tsname = '{}_{}.gz'.format(item['url'].replace('/','{}'),time.time())
        path = 'content/{}'.format('.'.join(extract(item['url'])))
        make_sure_path_exists(path)

        fname = '{}/{}'.format(path,tsname)
        with gzip.open(fname,'w') as f:
            f.write(item['content'].encode('utf-8'))

        return item
