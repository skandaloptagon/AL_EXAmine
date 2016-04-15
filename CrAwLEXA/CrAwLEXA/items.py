# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html
#
# Author: John Skandalakis

import scrapy

#Structure that contains URL data along with content from site
class CrawlexaItem(scrapy.Item):
    #Fields that we use in our struct, we have a URL which contains the actual URL of the site we have crawled along with the actual content of that URL.
    url = scrapy.Field()
    content = scrapy.Field()
    pass
