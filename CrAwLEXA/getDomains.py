# AUTHOR: JOHN SKANDALAKIS
# USE: This program queries the Alexa API to get the URLs associated with an 
# alexa category. To use this file you need an alexa api key file as provided 
# by AWS. The file needs to be located in the same directory


from awis import AwisApi
import os.path


# grab the api key and id from file and create the alexa object
ACCESS_ID = None
SECRET_ACCESS_KEY = None
with open("rootkey.csv","r") as keyfile:
    ACCESS_ID = keyfile.readline().split("=")[1].strip()
    SECRET_ACCESS_KEY = keyfile.readline().split("=")[1].strip()
api = AwisApi(ACCESS_ID, SECRET_ACCESS_KEY)

# check the list of categories you want to take from alexa
with open('categories.csv','r') as c:

    
    for category in c:
        category = category.strip()
        
        fname = "categories/"+category.replace('/','-')

        # check to make sure you haven't already 
        # done this so you don't spend money
        if os.path.isfile(fname):
            print fname, "already exists"
            continue        


        i = 1
        with open(fname,"w") as f:
            doit = True
            while doit:
                try:
                    print "alexa request",category,i
                    # make a query. 
                    tree = api.category_listings(category, Recursive=True, Start=i)
                    
                    # parse the query so that you only get the urls
                    for item in tree.findall("//{%s}DataUrl" % api.NS_PREFIXES["awis"]):
                        f.write(item.text+'\n')

                    # if there are no more queries to be made then quit.
                    if len(tree.findall("//{%s}DataUrl" % api.NS_PREFIXES["awis"])) < 100:
                        doit = False
                        break
                    i += 100
                except Exception as e:
                    doit = False
