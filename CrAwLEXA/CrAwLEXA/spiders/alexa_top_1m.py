# -*- coding: utf-8 -*-
#
# Author: John Skandalakis

import scrapy
from StringIO import StringIO
from zipfile import ZipFile
from urllib import urlopen
import requests, zipfile, StringIO
import csv
from CrAwLEXA.items import CrawlexaItem

#Struct that has a series of functions that refresh our alexa data set along with parsing out the URL and content from our crawled code
class AlexaTop1mSpider(scrapy.Spider):

    name = "alexa_top_1m"

    #Initialize start_urls (scrapy uses this object later
    start_urls = []
    
    def __init__(self,n=100):
        
        print "Requesting Alexa Top 1 Million..."

        #Get the most recent top 1 million from Alexa
        r = requests.get("http://s3.amazonaws.com/alexa-static/top-1m.csv.zip")
    
        print "Unzipping Alexa Top 1 Million..."

        #Unzip the top 1 million from memory
        with zipfile.ZipFile(StringIO.StringIO(r.content)) as z:
        
            #Open and read as CSV
            with z.open("top-1m.csv") as f:
                alexaCSV = csv.reader(f)
                for row in alexaCSV:
                    if int(row[0]) > int(n):
                        break
                    self.start_urls.append('http://' + row[1] + '/')

        print "Starting Crawl..."

    def parse(self, response):

        #Populate the item CrawelexaItem for use in pipelines
        item = CrawlexaItem()
        item['url'] = response.url
        item['content']= response.body
        
        self.logger.info('successful parse of %s',response.url)

        yield item
