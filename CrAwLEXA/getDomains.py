from awis import AwisApi
import os.path

ACCESS_ID = None
SECRET_ACCESS_KEY = None
with open("rootkey.csv","r") as keyfile:
    ACCESS_ID = keyfile.readline().split("=")[1].strip()
    SECRET_ACCESS_KEY = keyfile.readline().split("=")[1].strip()
api = AwisApi(ACCESS_ID, SECRET_ACCESS_KEY)

with open('categories.csv','r') as c:

    for category in c:
        category = category.strip()

        fname = "categories/"+category.replace('/','-')

        if os.path.isfile(fname):
            print fname, "already exists"
            continue        

        i = 1
        with open(fname,"w") as f:
            doit = True
            while doit:
                try:
                    print "alexa request",category,i
                    tree = api.category_listings(category, Recursive=True, Start=i)
                    for item in tree.findall("//{%s}DataUrl" % api.NS_PREFIXES["awis"]):
                        f.write(item.text+'\n')
                    if len(tree.findall("//{%s}DataUrl" % api.NS_PREFIXES["awis"])) < 100:
                        doit = False
                        break
                    i += 100
                except Exception as e:
                    doit = False
