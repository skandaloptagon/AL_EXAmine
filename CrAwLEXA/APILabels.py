import requests
import zipfile
import StringIO
import csv
import time
r = requests.get("http://s3.amazonaws.com/alexa-static/top-1m.csv.zip")
print "Unzipping Alexa Top 1 Million..."

start_urls = []

# Unzip the top 1 million from memory
with zipfile.ZipFile(StringIO.StringIO(r.content)) as z:
        
    # Open and read as CSV
    with z.open("top-1m.csv") as f:
        alexaCSV = csv.reader(f)
        for row in alexaCSV:
            #TODO limit the number of URLS to the top n
            start_urls.append(row[1])

print "recording labels"
i = 0
with open('labels/wot_'+str(time.time()),'wb') as f:
    f.write('{')
    for i in range(0,len(start_urls),100):
        f.write('"'+str(i)+'" : ')
        urls = '/'.join(start_urls[i:i+100])+'/'
        r = requests.get('http://api.mywot.com/0.4/public_link_json2?hosts='+urls+'&callback=process&key=5f521233275b6b1e3998be0452a765b1c73a904e')
        f.write(r.text.split('(')[1].split(')')[0]+',\n')
        i += 100
