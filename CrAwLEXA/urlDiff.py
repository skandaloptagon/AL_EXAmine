from lxml.html.diff import htmldiff
import glob
import gzip
import os
import random
import logging
import multiprocessing

# encoding=utf8  
import sys  

reload(sys)  
sys.setdefaultencoding('ISO-8859-1')

logging.basicConfig(level=logging.DEBUG)

# The worker.
def do_stuff(i=0):
    temp = glob.glob('content/*')
    size = len(temp)/6

    # break the glob into 6 parts. 1 for each process and choose the 1/6th 
    # associated with the process
    temp = temp[int(i*size):int((i+1)*size)]
    
    logging.info("Starting chunk " + str(int(i*size)))

    for domain in temp:

        # Ignore the timestamps and get the unique paths
        uniq_urls = set()

        # get the list of uniq urls in each domain
        for filename in glob.glob(domain+'/*'):
            uniq_urls.add('_'.join(filename.split('_')[:-1]))

        # iterate the uniq urls
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

                # this needs to be done in order.
                timestamps.sort()

                #more parsing
                temp1 = path + '_' + str(timestamps[0]) + '.gz'
                
                # interate the copies in order
                for i in timestamps[1:]:

                    # construct the filename
                    temp2 = path + '_' + str(i) + '.gz'
                    
                    # construct the output file name
                    filename = "diffs/"+path+'_'+str(i)+'_diff.gz'

                    # check make sure the output folder exists
                    dir = os.path.dirname(filename)
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    # Skip the ones that have already been diffed
                    if not os.path.isfile(filename):

                        # Open the output file
                        with open(filename,'w') as f:
                            try:
                                try:
                                    # this is the sauce.  All of the diffing and writing happens in this line.
                                    f.write(htmldiff(gzip.open(temp1).read(),gzip.open(temp2).read()))
                                    logging.debug("successful write "+filename)
                                except Exception:
                                    # sometimes it doesn't like the sauce so I created verde.
                                    f.write(unicode(htmldiff(gzip.open(temp1).read().decode('utf-8','ignore'),gzip.open(temp2).read().decode('utf-8','ignore'))).encode('utf-8','ignore'))
                                    logging.debug("transcode write "+filename)
                            except IOError as e:
                                logging.debug("missing file " + filename)
                            except AssertionError as e:
                                logging.debug("Assertion Error " + filename + " : " + e)
                    else:
                        logging.debug("already exists " + filename)
                    temp1 = temp2
            except TypeError as e:
                print e
            


if __name__ == '__main__':
    jobs = []
    for i in range(6):
        p = multiprocessing.Process(target=do_shit, args=(i,))
        jobs.append(p)
        p.start()
