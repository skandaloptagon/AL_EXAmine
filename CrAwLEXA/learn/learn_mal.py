from bs4 import BeautifulSoup
import re
import os.path
import glob
import gzip
import nltk
from nltk.corpus import stopwords # Import the stop word list
import cPickle as pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.cross_validation import KFold
from sklearn import svm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-V','--vectorizer',default="count",help="The vectorizer type. [count,hashing,tf_idf]")
parser.add_argument('-N','--ngram_range',default=(0,1),help="The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n such that min_n <= n <= max_n will be used.")
parser.add_argument('-Md','--max_df',default=1.0,help="When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.")
parser.add_argument('-md','--min_df',default=1,help="When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.")
parser.add_argument('-mf','--max_features',type=int,default=None,help="If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. This parameter is ignored if vocabulary is not None.")
parser.add_argument('-u','--use_idf',default="count",help="Enable inverse-document-frequency reweighting.")
parser.add_argument('-s','--smooth_idf',default="count",help="Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions.")
parser.add_argument('-S','--sublinear_tf',default="count",help="Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).")


parser.add_argument('-c','--classifier',default="rf",help="The classifier type. [random_forest,,tf_idf]")


parser.add_argument('-ne','--n_estimators',default=10, help="The number of trees in the forest.")
parser.add_argument('-mcf','--max_class_features',default=2000,help="The number of features to consider when looking for the best split: If int, then consider max_features features at each split. If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split. If auto, then max_features=sqrt(n_features). If sqrt, then max_features=sqrt(n_features) (same as auto). If log2, then max_features=log2(n_features). If None, then max_features=n_features.Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features. Note: this parameter is tree-specific.")


parser.add_argument('-d','--max_depth',default=None,help="The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples. Ignored if max_leaf_nodes is not None. Note: this parameter is tree-specific.")
parser.add_argument('-mss','--min_samples_split',default=2,help="The minimum number of samples required to split an internal node. Note: this parameter is tree-specific.")
parser.add_argument('-msl','--min_samples_leaf',default=1,help="The minimum number of samples in newly created leaves. A split is discarded if after the split, one of the leaves would contain less then min_samples_leaf samples. Note: this parameter is tree-specific.")
parser.add_argument('-mwfl','--min_weight_fraction_leaf',default=0,help="The minimum weighted fraction of the input samples required to be at a leaf node. Note: this parameter is tree-specific.")
parser.add_argument('-mln','--max_leaf_nodes',default=None,help="Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes. If not None then max_depth will be ignored. Note: this parameter is tree-specific.")
parser.add_argument('-nj','--n_jobs',default=1,help="The number of jobs to run in parallel for both fit and predict. If -1, then the number of jobs is set to the number of cores.")

args = parser.parse_args()



# From https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    try:
        review_text = BeautifulSoup(raw_review,'html.parser').body.get_text() 
    except:
        review_text = BeautifulSoup(raw_review,'html.parser').get_text()
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))


known_labels = dict()
sector_id = dict()
train_dict = dict()
sector_dict = dict()

clean_train_reviews = []
train_sector = []
j= 2000
i = j

if not os.path.isfile("train_temp.p"):
    print "buildin model data"
    sector = 0
    for category in glob.glob('categories/*'):
        print category
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
                        with gzip.open(path,'r') as source:
                            try:
                                train_dict[domain]=review_to_words(source.read())
                                sector_dict[domain]=sector
                                #clean_train_reviews.append(review_to_words(source.read()))
                                #train_sector.append(sector)
                                break
                            except Exception as e:
                                print e
                                continue
        sector+=1
    
    print "adjusting model data"
    for domain in known_labels:
        if len(known_labels[domain]) > 1:
            continue
        try:
            clean_train_reviews.append(train_dict[domain])
            train_sector.append(sector_dict[domain])
            del train_dict[domain]
            del sector_dict[domain]
        except KeyError as ke:
            print "trying to access deleted data"

    pickle.dump([clean_train_reviews,train_sector,sector_id],open("train_temp.p",'wb'))
else:
    temp = pickle.load( open( "train_temp.p", "rb" ) )
    clean_train_reviews = temp[0]
    train_sector = temp[1]
    sector_id = temp[2]
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
                             #ngram_range = args.ngram_range, \
                             max_df = args.max_df, \
                             min_df = args.min_df )

elif(args.vectorizer.lower() == "hashing"):
    vectorizer = HashingVectorizer(decode_error='strict',\
                               n_features=args.max_features, \
                               non_negative=True, \
                               #ngram_range=args.ngram_range, \
                               max_df = args.max_df, \
                               min_df = args.min_df )

elif(args.vectorizer.lower() == "tf_idf"):
    vectorizer = TfidfVectorizer(analyzer="word", \
                             max_features = args.max_features,\
                             #ngram_range = args.ngram_range,\
                             max_df = args.max_df,\
                             min_df = args.min_df )

classifier = None

# Initialize a Random Forest classifier with 100 trees
if args.classifier.lower() =="rf":
    classifier = RandomForestClassifier(n_jobs=-1,n_estimators = 400) 
elif args.classifier.lower() == "kn":
    classifier = KNeighborsClassifier(n_neighbors = 28,weights='distance', n_jobs = -1)
elif args.classifier.lower() == "svc":
    classifier = svm.SVC()

j=0
kf = KFold(len(clean_train_reviews), n_folds=10, shuffle=True)

auc_sums = dict()

for i in range(0,48):
    auc_sums[i] = 0

for train_index, test_index in kf:

    print "beginning fold",j
    j+= 1

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in train_index:
        X_train.append(clean_train_reviews[i])
        y_train.append(train_sector[i])
    for i in test_index:
        X_test.append(clean_train_reviews[i])
        y_test.append(train_sector[i])

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
    for i in range(0,len(sector_id)):
        fpr, tpr, thresholds = roc_curve(y_test, result, pos_label=i)
        roc_auc = auc(fpr, tpr)
        try:
            temp_auc+=roc_auc
            auc_sums[i] += roc_auc
        except:
            pass

        print '\troc_auc:{}\tsector:{}'.format(roc_auc,sector_id[i])
    print "Average auc",temp_auc/len(sector_id)
    print "Accuracy", accuracy_score(y_test, result)


