import glob
import os.path

for g in glob.glob("categories/*"):
    with open(g,'r') as f:
        for url in f:
            url = url.split('/')[2]
            print url,os.path.isdir('content/'+url+'/')
