from __future__ import division

from bs4 import BeautifulSoup
import re
import os.path
import glob
import gzip
import nltk
from nltk.corpus import stopwords # Import the stop word list
import cPickle as pickle
import HTMLParser

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.cross_validation import KFold
from sklearn import svm


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i',default="temp.p",help="input pickle file name")
parser.add_argument('-V','--vectorizer',default="count",help="The vectorizer type. [count,hashing,tf_idf]")
parser.add_argument('-N','--ngram_range',default=(0,1),help="The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.")
parser.add_argument('-Md','--max_df',type=float,default=1.0,help="When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.")
parser.add_argument('-md','--min_df',type=float,default=1,help="When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.")
parser.add_argument('-mf','--max_features',type=int,default=None,help="If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. This parameter is ignored if vocabulary is not None.")
parser.add_argument('-norm',type=str,default=None,help="Norm used to normalize term vectors. None for no normalization.")
parser.add_argument('-u','--use_idf',type=bool,default=True,help="Enable inverse-document-frequency reweighting.")
parser.add_argument('-s','--smooth_idf',type=bool,default=True,help="Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.")
parser.add_argument('-S','--sublinear_tf',type=bool,default=False,help="Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).")


parser.add_argument('-c','--classifier',default="rf",help="The classifier type. [random_forest,,tf_idf]")


parser.add_argument('-ne','--n_estimators',default=10, help="The number of trees in the forest.")
parser.add_argument('-n_neighbors',default=10,type=int, help="Number of neighbors to use by default for k_neighbors queries.")
parser.add_argument('-mcf','--max_class_features',default=2000,help="The number of features to consider when looking for the best split: If int, then consider max_features features at each split. If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split. If auto, then max_features=sqrt(n_features). If sqrt, then max_features=sqrt(n_features) (same as auto). If log2, then max_features=log2(n_features). If None, then max_features=n_features.Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features. Note: this parameter is tree-specific.")


parser.add_argument('-d','--max_depth',default=None,help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. Ignored if max_leaf_nodes is not None. Note: this parameter is tree-specific.")
parser.add_argument('-mss','--min_samples_split',default=2,help="The minimum number of samples required to split an internal node. Note: this parameter is tree-specific.")
parser.add_argument('-msl','--min_samples_leaf',default=1,help="The minimum number of samples in newly created leaves. A split is discarded if after the split, one of the leaves would contain less then min_samples_leaf samples. Note: this parameter is tree-specific.")
parser.add_argument('-mwfl','--min_weight_fraction_leaf',default=0,help="The minimum weighted fraction of the input samples required to be at a leaf node. Note: this parameter is tree-specific.")
parser.add_argument('-mln','--max_leaf_nodes',default=None,help="Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes. If not None then max_depth will be ignored. Note: this parameter is tree-specific.")
parser.add_argument('-nj','--n_jobs',default=1,help="The number of jobs to run in parallel for both fit and predict. If -1, then the number of jobs is set to the number of cores.")
parser.add_argument('-f','--fig',default="temp.png",help="the name of the figure")

args = parser.parse_args()

def parse_tags(source):
    soup = BeautifulSoup(source, 'html.parser')

    linkwords = list()
    links = list()
    iframes = list()
    for link in soup.find_all('a'):
        link = link.get('href')
        if link == None:
            continue
        if link.startswith('http'):
            try:
                links.append(str(link).split('/')[2])
                linkwords.append(' '.join(re.split('/|-|_',str(link))[3:-1]))
            except UnicodeEncodeError as e:
                pass
            except IndexError as e:
                links.append('local')

    for iframe in soup.find_all('iframe')[:-1]:
        if iframe == None:
            continue
        try:
            iframes.append(iframe.get('src').split('/')[2])
        except UnicodeEncodeError as e:
            pass
        except AttributeError as e:
            pass 
        except IndexError as e:
            iframes.append('local')
 
    letters_only = re.sub("[^a-zA-Z]", " ", soup.get_text()) 

    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
        

    try:
        linkwords = ' '.join(linkwords)
    except TypeError as e:
        linkwords = ''

    try:
        words = ' '.join(meaningful_words)
    except TypeError as e:
        words = ''

    try:
        links = ' '.join(links)
    except TypeError as e:
        links = ''

    try:
        iframes = ' '.join(iframes)
    except TypeError as e:
        iframes = ''
    
    return links, iframes, words, linkwords


known_labels = dict()
sector_id = dict()
train_dict = dict()
sector_dict = dict()

words = []
links =[]
iframes = []
linkwords = []

train_sector = []
j= 10000
i = j

if not os.path.isfile(args.i):
    print "Building model data..."
    sector = 0
    for category in glob.glob('categories/*'):
        print "\tGround Truth:",category
        with open(category, 'r') as f:
            sector_id[sector]=category
            for domain in f:
                domain = domain.split('/')[2]
                if i <= 0:
                    i = j
                    break
                i -= 1
                if os.path.isdir('content/'+domain+'/'):

                    if domain not in known_labels:
                        known_labels[domain] = list([category])
                    else:
                        known_labels[domain].append(category)

                    for path in glob.glob('content/'+domain+'/*'):
                        try:
                            with gzip.open(path,'r') as source:
                                train_dict[domain]=parse_tags(source.read())
                                sector_dict[domain]=sector
                                break
                        except HTMLParser.HTMLParseError as e:
                            pass
        sector+=1
    
    print "Adjusting model data..."
    for domain in known_labels:
        if len(known_labels[domain]) > 1:
            continue
        try:
            links.append(train_dict[domain][0])
            iframes.append(train_dict[domain][1])
            words.append(train_dict[domain][2])
            linkwords.append(train_dict[domain][3])
            train_sector.append(sector_dict[domain])
            del train_dict[domain]
            del sector_dict[domain]
        except KeyError as ke:
            print "Trying to access deleted data..."

    pickle.dump([links,iframes,words, train_sector, train_sector, sector_id],open(args.i,'wb'))
else:
    temp = pickle.load(open( args.i, "rb" ))
    links = temp[0]
    iframes = temp[1]
    words = temp[2]
    linkwords = temp[3]
    train_sector = temp[4]
    sector_id = temp[5]
    temp = None

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  

vectorizer = None

if (args.vectorizer.lower() == "count"):
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = args.max_features, \
                             ngram_range = (1,2), \
                             max_df = args.max_df, \
                             min_df = args.min_df )

elif(args.vectorizer.lower() == "hashing"):
    vectorizer = HashingVectorizer(decode_error='strict',\
                               n_features=args.max_features, \
                               non_negative=True,\
                                norm = args.norm )

elif(args.vectorizer.lower() == "tf_idf"):
    vectorizer = TfidfVectorizer(analyzer="word", \
                             max_features = args.max_features,\
                             #ngram_range = args.ngram_range,\
                             max_df = args.max_df,\
                             min_df = args.min_df,\
                             use_idf = args.use_idf,\
                             smooth_idf = args.smooth_idf,\
                             sublinear_tf = args.sublinear_tf, \
                             norm = args.norm )

if_vectorizer = TfidfVectorizer(analyzer="word",max_features = 200)
lw_vectorizer = CountVectorizer(analyzer="word",max_features = args.max_features)
l_vectorizer = TfidfVectorizer(analyzer="word", \
                             max_features = args.max_features,\
                             #ngram_range = args.ngram_range,\
                             max_df = args.max_df,\
                             min_df = args.min_df,\
                             use_idf = args.use_idf,\
                             smooth_idf = args.smooth_idf,\
                             sublinear_tf = args.sublinear_tf, \
                             norm = args.norm )
classifier = None

# Initialize a Random Forest classifier with 100 trees
if args.classifier.lower() =="rf":
    classifier = RandomForestClassifier(n_jobs=-1,n_estimators = 400) 
elif args.classifier.lower() == "kn":
    classifier = KNeighborsClassifier(n_neighbors = args.n_neighbors, weights='distance', n_jobs = -1)
elif args.classifier.lower() == "svc":
    classifier = svm.SVC()

roc_aucL = dict()
fprL = [[[] for x in range(10)] for x in range(len(sector_id))] 
tprL = [[[] for x in range(10)] for x in range(len(sector_id))] 
for i in range(len(sector_id)):
    roc_aucL[i] = 0

j=0
kf = KFold(len(links), n_folds=10, shuffle=True)
for train_index, test_index in kf:

    print "beginning fold",j
    j+= 1

    U_train = []
    V_train = []
    W_train = []
    X_train = []
    y_train = []

    U_test = []
    V_test = []
    W_test = []
    X_test = []
    y_test = []

    for i in train_index:
        U_train.append(linkwords[i])
        V_train.append(links[i])
        W_train.append(iframes[i])
        X_train.append(words[i])
        y_train.append(train_sector[i])
    for i in test_index:
        U_test.append(linkwords[i])
        V_test.append(links[i])
        W_test.append(iframes[i])
        X_test.append(words[i])
        y_test.append(train_sector[i])

    print "transforming train vector Y..."
    U_train = lw_vectorizer.fit_transform(U_train)
    U_train = U_train.toarray()
    print "transforming train vector Y..."
    V_train = l_vectorizer.fit_transform(V_train)
    V_train = V_train.toarray()
    
    print "transforming train vector W..."
    W_train = if_vectorizer.fit_transform(W_train)
    W_train = W_train.toarray()
    
    print "transforming train vector X..."
    X_train = vectorizer.fit_transform(X_train)
    X_train = X_train.toarray()
    print V_train.shape,W_train.shape,X_train.shape  
    print "stacking arrays"
    X_train = np.hstack((U_train, V_train, W_train, X_train ))
    print X_train.shape

    print "fitting the classifier..."
    classifier = classifier.fit( X_train, y_train )
 
    print "transforming test vector Y..."
    U_test = lw_vectorizer.transform(U_test)
    U_test = U_test.toarray() 

    print "transforming test vector Y..."
    V_test = l_vectorizer.transform(V_test)
    V_test = V_test.toarray() 

    print "transforming test vector W..."
    W_test = if_vectorizer.transform(W_test)
    W_test = W_test.toarray() 

    print "transforming test vector X..."
    X_test = vectorizer.transform(X_test)
    X_test = X_test.toarray() 

    X_test = np.hstack((U_test, V_test, W_test, X_test ))
    print X_test.shape


    print "predicting..."
    result = classifier.predict(X_test)

    print "results..." 
    temp_auc = 0
    
    for i in range(0,len(sector_id)):
        fpr, tpr, thresholds = roc_curve(y_test, result, pos_label=i)
        roc_auc = auc(fpr, tpr)

        
        tprL[i][j-1] = tpr.tolist()
        fprL[i][j-1] = fpr.tolist()
        roc_aucL[i] += roc_auc


        print '\troc_auc:{}\tsector:{}'.format(roc_auc,sector_id[i])


# PART 2 - THE PLOTTENING
fig = plt.figure(num=None, figsize=(16, 16), dpi=256, facecolor='w', edgecolor='k')

for i in range(0,len(sector_id)):
    fprA=[sum(e)/len(e) for e in zip(*fprL[i])]
    tprA=[sum(e)/len(e) for e in zip(*tprL[i])]
    plt.plot(fprA, tprA, label='{} (area = {})'.format(' '.join(sector_id[i].split('-')[2:]),float(roc_aucL[i]/10)))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(args.fig)
