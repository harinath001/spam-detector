from textblob import TextBlob
import pandas as pd
import csv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
# from sklearn.grid_search import GridSearchCV
# from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
# from sklearn.learning_curve import learning_curve



df = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])
english_stop_words = stopwords.words('english')
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
unwanted_chars = re.compile("[^0-9a-zA-Z ]+", re.IGNORECASE)

def lemmatize(word):
    try:
        return wordnet_lemmatizer.lemmatize(str(word.encode('ascii', 'ignore')))
    except Exception as ex:
        print word
        return ""


def stem(word):
    try:
        return porter_stemmer.stem(str(word.encode('ascii', 'ignore')))
    except Exception as ex:
        print word
        return ""


def remove_stop_words(word_list):
    final_list = []
    for each_word in word_list:
        if each_word not in english_stop_words:
            final_list += [each_word]
    return final_list

def split_into_lemmas(message):
    return remove_stop_words([lemmatize(stem(each_word)) for each_word in
                       word_tokenize(unwanted_chars.sub("", unicode(message, 'utf8')))])

df['words_list'] = df.message.apply(split_into_lemmas)  # if each_word.encode('ascii','ignore') not in english_stop_words])
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(df["message"])
messages_bow = bow_transformer.transform(df['message'])
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
spam_detector = MultinomialNB().fit(messages_tfidf, df['label'])



bow = bow_transformer.transform(["hello this is from customer care you can easily get the loan"])
tfidf = tfidf_transformer.transform(bow)
print spam_detector.predict(tfidf)[0]