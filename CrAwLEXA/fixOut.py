# AUTHOR: JOHN SKANDALAKIS
# DEPRECATED
# This program runs through the database and fixes the Encoding so they all match.

from lxml.html.diff import htmldiff
import glob
import gzip
import os

    
# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('ISO-8859-1')

for domain in glob.glob('content/*'):

    for filename in glob.glob(domain+'/*'):
        
        print "Fixing:",filename
        try:
            with gzip.open(filename,'r') as f:
            

    temp = unicode(f.read().decode('utf-8','ignore')).encode('utf-8','ignore')
            with gzip.open(filename,'w') as f:
                f.write(temp)
        except IOError as e:
            print "IOError:",filename

