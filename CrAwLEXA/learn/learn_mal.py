from bs4 import BeautifulSoup
import re
import os.path
from glob import glob
import gzip
import nltk
from nltk.corpus import stopwords # Import the stop word list
import cPickle as pickle
import requests
import zipfile
import StringIO
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.cross_validation import KFold
from sklearn import svm

from lxml.html.diff import htmldiff

import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('-i','--inp',default="in.p",help="The input pickle file. If it does not exist it will be created")
parser.add_argument('-o','--out',default="images/temp.png",help="The output graph image")

args = parser.parse_args()

cl = []
features = []

mal_urls = []
ben_urls = []


def extract_features(source1,source2):
    source1 = gzip.open(source1).read()
    source2 = gzip.open(source2).read()
    soup = BeautifulSoup(htmldiff(source1,source2),'html.parser')
    inserts = []
    for ins in soup.find_all('ins'):
        ins = str(ins)[5:-6].strip() 
        if '<' in ins and '>' in ins:
            if not ins.startswith('<'):
                ins = ins[ins.index('<'):]
            if not ins.endswith('>'):
                ins = ins[:ins.index('>')]

            ins = re.sub(r'\s+', '', ins)
            ins = re.sub(r'".*"', '""', ins)
            ins = re.sub(r'>.*<','><', ins)
            inserts.append(ins)

    return ' '.join(inserts)


if not os.path.isfile(args.inp):
    print "buildin model data"

    # Add features for blacklisted sites
    with open('/nethome/akountouras3/crawl-resources-bl-with-source-and-time.tsv','r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.split()

            tempts = None            
            try:
                tempts = int(line[2])
            except:
                continue

            state = False
            gl = glob('content/'+line[1].split('/')[2]+'/*')
            tracker = 0


            for path in gl:
                ts = float(path.split('_')[-1].split('.')[0])
                tracker += 1
                if tempts < ts:
                    gl = gl[tracker-1:tracker+1]
                    features.append(extract_features(gl[0],gl[-1]))
                    cl.append(0)
                    break
     
    # Add features for Okay sites
    r = requests.get("http://s3.amazonaws.com/alexa-static/top-1m.csv.zip")
    with zipfile.ZipFile(StringIO.StringIO(r.content)) as z:
        with z.open("top-1m.csv") as f:
            alexaCSV = csv.reader(f)
            for row in alexaCSV:
                if int(row[0]) > 1000:
                    break
                files = glob('content/www.' + row[1] + '/*')[0:2]
                if len(files) == 2:
                    try:
                        features.append(extract_features(files[0],files[-1]))
                        cl.append(1)
                    except IOError as e:
                        pass



    pickle.dump([cl,features],open(args.inp,'wb'))
else:
    temp = pickle.load( open( args.inp, "rb" ) )
    cl = temp[0]
    features = temp[1]
    temp = None

vectorizer = TfidfVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 20000)

classifier = KNeighborsClassifier(n_jobs=-1, n_neighbors=3)

j=0
kf = KFold(len(cl), n_folds=10, shuffle=True)

numbers = []

for train_index, test_index in kf:

    print "beginning fold",j
    j+= 1

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in train_index:
        y_train.append(cl[i])
        X_train.append(features[i])
    for i in test_index:
        y_test.append(cl[i])
        X_test.append(features[i])

    print "transforming train vector..."
    train_data_features = vectorizer.fit_transform(X_train)
    train_data_features = train_data_features.toarray()

    print "fitting the classifier..."
    classifier = classifier.fit( train_data_features, y_train )
 
    print "transforming test vector..."
    test_data_features = vectorizer.transform(X_test)
    test_data_features = test_data_features.toarray() 

    print "predicting..."
    result = classifier.predict(test_data_features)

    print "results..." 
    temp_auc = 0
    
    fpr, tpr, thresholds = roc_curve(y_test, result)
    roc_auc = auc(fpr, tpr)
    

    numbers.append([roc_auc,fpr,tpr])

# PART 2 - THE PLOTTENING
fig = plt.figure(num=None, figsize=(16, 16), dpi=256, facecolor='w', edgecolor='k')

for i in range(0,len(numbers)):
    plt.plot(numbers[i][1], numbers[i][2], label='{} (area = {})'.format("fold "+ str(i),numbers[i][0]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(args.out)
