# -*- coding: utf-8 -*-
#
# Author: John Skandalakis

import scrapy
from StringIO import StringIO
from CrAwLEXA.items import CrawlexaItem
import glob, os.path

class AlexaTop1mSpider(scrapy.Spider):

    name = "alexa_top_c"

    # initialize start_urls (scrapy uses this object later
    start_urls = []
    
    def __init__(self,n=100):
    
        print "Building start list"

        for g in glob.glob("categories/*"):
            with open(g,'r') as f:
                for url in f:
                    self.start_urls.append(url.strip())

        print "Starting Crawl..."

    def parse(self, response):

        #Populate the item for use in pipelines
        item = CrawlexaItem()
        item['url'] = response.url
        item['content']= response.body
        
        self.logger.info('successful parse of %s',response.url)

        yield item
