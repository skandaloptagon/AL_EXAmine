import glob
import re
import gzip
import nltk
import os.path
from bs4 import BeautifulSoup

def parse_tags(source):
    soup = BeautifulSoup(source, 'html.parser')
    print soup.prettify()
    links = set()
    iframes = set()
    for link in soup.find_all('a'):
        link = link.get('href')
        if link == None:
            continue
        if link.startswith('http'):
            try:
                links.add(str(link).split('/')[2])
            except UnicodeEncodeError as e:
                pass

    for iframe in soup.find_all('iframe')[:-1]:
        if link == None:
            continue
        try:
            iframes.add(iframe.get('src').split('/')[2])
        except UnicodeEncodeError as e:
            pass
 
    try:
        return ' '.join(links),' '.join(iframes)
    except TypeError as e:
        return ''
er=0
for category in glob.glob('categories/*'):
    print category
    with open(category, 'r') as f:
        for domain in f:
            domain = domain.split('/')[2]
            if os.path.isdir('content/'+domain+'/'):
                for path in glob.glob('content/'+domain+'/*'):
                    print path
                    with gzip.open(path,'r') as source:
                        print parse_tags(source.read())
                        break
                        '''except Exception as e:
                            er +=1
                            if er >= 3:
                                er = 0
                                break
                            print e
                            continue
                        '''
