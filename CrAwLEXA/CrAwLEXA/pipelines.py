# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
#
# Author: John Skandalakis

import umsgpack

class MessagePackPipeline(object):

    # open content.bin
    f = open('content.bin', 'wb')

    def process_item(self, item, spider):
        # pack an object consisting of url and the content of the url
        umsgpack.pack({item['url']:item['content']}, self.f)
        return item

    def close_spider(self, spider):
        # close the file for good measure
        self.f.close()
