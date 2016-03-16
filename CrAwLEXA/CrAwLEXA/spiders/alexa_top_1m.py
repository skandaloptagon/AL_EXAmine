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

class AlexaTop1mSpider(scrapy.Spider):
    name = "alexa_top_1m"

    # initialize start_urls (scrapy uses this object later
    start_urls = []

    print "Requesting Alexa Top 1 Million..."

    # get the most recent top 1 million from Alexa
    r = requests.get("http://s3.amazonaws.com/alexa-static/top-1m.csv.zip")
    
    print "Unzipping Alexa Top 1 Million..."

    # Unzip the top 1 million from memory
    with zipfile.ZipFile(StringIO.StringIO(r.content)) as z:
        
        # Open and read as CSV
        with z.open("top-1m.csv") as f:
            alexaCSV = csv.reader(f)
            for row in alexaCSV:
                #TODO limit the number of URLS to the top n
                start_urls.append('http://' + row[1] + '/')


    def parse(self, response):

        #Populate the item for use in pipelines
        item = CrawlexaItem()
        item['url'] = response.url
        item['content'] = response.body_as_unicode()
        yield item
