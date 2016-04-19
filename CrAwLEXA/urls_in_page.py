# AUTHOR: JOHN SKANDALAKIS
# The purpose of this code is to parse through every crawled page as 
# effiently as possible and to collect the URLs and print the URLs, 
# a timestamp and the crawled urls line by line

from glob import glob
import gzip
from multiprocessing import Process, Queue, Lock

from bs4 import BeautifulSoup
import time
import datetime
import logging
logging.basicConfig(filename='urls_in_page.log',level=logging.DEBUG)

# this helper function just extracts urls from the a tags and iframes in the 
# html source code
def extract_urls(source):
    soup = BeautifulSoup(source, 'html.parser')
    urls = set()
    
    # iterate through atags and iframes.
    for url in soup.find_all(['a','iframe']):
        href = url.get('href')
        src = url.get('src')
        
        # it's either an a tag or an iframe. only one way to find out.
        if src != None:
            urls.add(src)
        elif href != None:
            urls.add(href)
    
    return urls

# Add paths to the queue to be used by the other processes
def accumulator(q):
    for domain in glob('content/*'):
        for url in glob(domain + '/*'):
            q.put(url)

# This pulls the url path from the queue and parses it for information and gets
# the links from the source and prints a tsv
def worker(l,q):

    # as long as there are still items in the queue keep running
    while q:

        path = q.get()

        # unzip the file and extract the urls from the sourc
        with gzip.open(path,'r') as f:

            # get and parse the file path for info
            path = path.split('/')[-1]
            ts = '.'.join(path.split('_')[-1].split('.')[0:-1])
            crawled_url = path.split('/')[-1].split('_')[0].replace('{}','/')

            for uniq_url in extract_urls(f.read()):
                try: 
                    print uniq_url, ts, crawled_url
                except UnicodeEncodeError as uee:
                    logging.debug('UnicodeEncodeError\t{}\tin file {}'.format(uee,path))
            

if __name__== '__main__':
    jobs = []
    q = Queue()
    l = Lock()

    # add the accumulator for paths.
    p = Process(target=accumulator,args=(q,))
    jobs.append(p)
    p.start()

    # 6 cores one process already started add 5 and we're full!
    for i in range(5):
        p = Process(target=worker,args=(l,q))
        jobs.append(p)
        p.start()
