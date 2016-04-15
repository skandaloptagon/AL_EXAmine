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
import logging

#Function call that attempts to create a directory (assuming it hasn't already been created). It takes in the path as a param.
def make_sure_path_exists(path):
    try:
        os.makedirs(path) #Try to create the directory
    except OSError as exception: #Throw error if the directory already exists
        if exception.errno != errno.EEXIST:
            raise

#Struct that processes scrapped URL items and places in the crawled data into their unique zip files.
class MessagePackPipeline(object):
    #Function call that creates the zip files (with unique timestamps in the file names) with the specfically crawled data.
    def process_item(self, item, spider):
        #Construct our file name based upon the URL and 'when' crawled timestamp
        tsname = '{}_{}.gz'.format(item['url'].replace('/','{}'),time.time())
        #We create a path directory based upon the URLs name.
        path = 'content/{}'.format('.'.join(extract(item['url'])))
        make_sure_path_exists(path)

        #Create teh file and write it to said file using the gzip command.
        fname = '{}/{}'.format(path,tsname)
        with gzip.open(fname,'w') as f:
            f.write(item['content'])

        #report back a successful write to our log file.
        logging.info("Successful write of %s", item['url'])
        return item
