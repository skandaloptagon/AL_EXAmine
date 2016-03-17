import json as simplejson
import urllib
import urllib2
import ssl

scanner = "http://www.punkspider.org/service/search/detail/"
url = "www.google.com/fjofowefjo"
parsedURL = url.split('.')
rURL = ''
for x in reversed(parsedURL):
    rURL = rURL + '.' + x

rURL = rURL[1:]
print rURL
parameters = {"url": rURL}
data = urllib.urlencode(parameters)
req = urllib2.Request(scanner, data)

context = ssl._create_unverified_context()
response = urllib2.urlopen(req)
json = response.read()
print json
