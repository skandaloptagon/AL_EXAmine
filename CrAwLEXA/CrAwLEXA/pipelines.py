# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import umsgpack

class MessagePackPipeline(object):
    f = open('content.bin', 'wb')

    def process_item(self, item, spider):
        umsgpack.pack({item['url']:item['content']}, self.f)
        return item

    def close_spider(self, spider):
        self.f.close()
