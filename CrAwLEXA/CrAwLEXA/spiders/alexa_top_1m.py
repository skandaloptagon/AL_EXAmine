# -*- coding: utf-8 -*-
import scrapy
from StringIO import StringIO
from zipfile import ZipFile
from urllib import urlopen
import requests, zipfile, StringIO
import csv
from CrAwLEXA.items import CrawlexaItem

class AlexaTop1mSpider(scrapy.Spider):
    name = "alexa_top_1m"

    start_urls = []

    print "Requesting Alexa Top 1 Million..."
    r = requests.get("http://s3.amazonaws.com/alexa-static/top-1m.csv.zip")
    print "Unzipping Alexa Top 1 Million..."
    with zipfile.ZipFile(StringIO.StringIO(r.content)) as z:
        with z.open("top-1m.csv") as f:
            alexaCSV = csv.reader(f)
            for row in alexaCSV:
                start_urls.append('http://' + row[1] + '/')


    def parse(self, response):
        item = CrawlexaItem()
        item['url'] = response.url
        item['content'] = response.body
        yield item
