from lxml.html.diff import htmldiff
import glob
import gzip
import os
import random
import logging
    
# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('ISO-8859-1')


logging.basicConfig(level=logging.DEBUG)
temp = glob.glob('content/*')

random.shuffle(temp)

for domain in temp:

    # Ignore the timestamps and get the unique paths
    uniq_urls = set()
    for filename in glob.glob(domain+'/*'):
        uniq_urls.add('_'.join(filename.split('_')[:-1]))

    for path in uniq_urls:

        # Get the timestamps for each uniq filename
        timestamps = set()
        for filename in glob.glob(path + '*'):
            try:
                timestamps.add(float('.'.join(filename.split('_')[-1].split('.')[:-1])))
            except Exception as e:
                pass
        try:
            timestamps = list(timestamps)
            timestamps.sort()
            temp1 = path + '_' + str(timestamps[0]) + '.gz'
            for i in timestamps[1:]:
                temp2 = path + '_' + str(i) + '.gz'
                filename = "diffs/"+path+'_'+str(i)+'_diff.gz'
                dir = os.path.dirname(filename)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                # Skip the ones that have already been diffed
                if not os.path.isfile(filename):
                    with open(filename,'w') as f:
                        try:
                            try:
                                f.write(htmldiff(gzip.open(temp1).read(),gzip.open(temp2).read()))
                                logging.debug("successful write "+filename)
                            except Exception:
                                f.write(unicode(htmldiff(gzip.open(temp1).read().decode('utf-8','ignore'),gzip.open(temp2).read().decode('utf-8','ignore'))).encode('utf-8','ignore'))
                                logging.debug("transcode write "+filename)
                        except IOError as e:
                            logging.debug("missing file " + filename)
                else:
                    logging.debug("already exists " + filename)
                temp1 = temp2
        except TypeError as e:
            print e
        
