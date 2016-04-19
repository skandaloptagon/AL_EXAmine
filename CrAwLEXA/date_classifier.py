import os
import subprocess
import glob
import time
import datetime

#This piece of code just takes in the filenames of the crawled URLs and parses out the time stamps along with the appropriate URL.

#First for loop gets the list of all of the domain directories from the content directory.
for domain in glob.glob('content/*'):
    #Second loop iterates through the files in each of the crawled domains to extract out specifc timestamps.
    for filename in glob.glob(domain + '/*'):
        #Parse filename by underscore and print out the URL along with the converted timestamp time (from epoch to traditional time).
        filename = filename.split('_')
        print filename[0].split('/')[-1].replace('{}','/'),time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(float('.'.join(filename[-1].split('.')[0:1]))))
