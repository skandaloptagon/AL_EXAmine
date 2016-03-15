from lxml.html.diff import htmldiff
import glob
import gzip
import os


for domain in glob.glob('content/*'):

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
                print e, filename
        
        # We really only care about the oldest and the newest
        mn = None
        mx = None        

        try:
            mn = min(timestamps)
            mx = max(timestamps)
        except Exception as e:
            continue        

        # Make sure you are taking fresh diffs
        if mn != mx:
            
            # get the oldest and the newest recorded page. 
            # Theoretically there should only ever be two pages to compare
            try:
                old_html = gzip.open(path+'_'+str(mn)+'.gz').read()
                new_html = gzip.open(path+'_'+str(mx)+'.gz').read()
            except Exception as e:
                print e, path+'_'+str(mn)+'.gz'
                continue

            # write the diff to a file
            with gzip.open(path+'_'+str(mn)+'diff'+'.gz','w') as f:
                try:
                    f.write(htmldiff(old_html,new_html).encode('ascii', 'replace'))
                except AssertionError as e:
                    print e, 

            # delete all old files
            for i in timestamps-set([mx]):
                try:
                    os.remove(path+'_'+str(i)+'.gz')
                except Exception as e:
                    print e, path+'_'+str(i)+'.gz'
