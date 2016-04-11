from bs4 import BeautifulSoup
import re
import os.path
import glob
import gzip
import nltk
from nltk.corpus import stopwords # Import the stop word list
import cPickle as pickle

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import KFold

# From https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review,'lxml').get_text()
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

clean_train_reviews = []
train_sector = []

if not os.path.isfile("train.p"):

    sector = 0
    for category in glob.glob('categories/*'):
        print category
        with open(category, 'r') as f:
            for domain in f:
                domain = domain.split('/')[2]
                if os.path.isdir('content/'+domain+'/'):

                    if domain not in known_labels:
                        known_labels[domain] = list([category])
                    else:
                        known_labels[domain].append(category)

                    for path in glob.glob('content/'+domain+'/*'):
                        with gzip.open(path,'r') as source:
                            try:
                                clean_train_reviews.append(review_to_words(source.read()))
                                train_sector.append(sector)
                                break
                            except Exception as e:
                                print e
                                continue
        sector+=1

    pickle.dump((clean_train_reviews,train_sector),open("train.p",'wb'))
else:
    temp = pickle.load( open( "train.p", "rb" ) )
    clean_train_reviews = temp[0]
    train_sector = temp[1]
    temp = None

# Initialize the "HashingVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = HashingVectorizer(decode_error='strict',
                               n_features=2 ** 15,
                               non_negative=True)


# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

j=0
kf = KFold(len(clean_train_reviews), n_folds=10, shuffle=True)
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
    
    forest = forest.fit( train_data_features, y_train )
 
    print "transforming test vector..."
    test_data_features = vectorizer.transform(X_test)
    test_data_features = test_data_features.toarray() 

    print "predicting..."
    result = forest.predict(test_data_features)


    print "results..."
    for i in range(0,48):
        fpr, tpr, thresholds = roc_curve(y_test, result, pos_label=i)
        #print fpr,tpr,thresholds
        roc_auc = auc(fpr, tpr)
        try:
            auc_sums[i] += roc_auc
        except:
            print ''
        print '\t',i, roc_auc
print "Average results:"
for i in auc_sums:
    print i, auc_sums[i]/10
