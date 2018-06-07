# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pycrfsuite
from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import nltk
nltk.download('averaged_perceptron_tagger')



with codecs.open("shopi_consolidated_data.xml", "r", "utf-8") as infile:
    soup = bs(infile, "html5lib")

docs = []
for elem in soup.find_all("document"):
    texts = []

    
    for c in elem.find("textwithnamedentities").children:
        if type(c) == Tag:
            if c.name == "num":
                label = "N" 
            elif c.name == "unit":
                label = "U"
            elif c.name == "product":
                label = "P"
            elif c.name == "other":
                label = "O"
            elif c.name == "company":
                label = "C"
            elif c.name == "shop":
                label = "S"
            else:
                label = "Op"  # irrelevant word
            for w in c.text.split(" "):
                if len(w) > 0:
                    texts.append((w, label))
    docs.append(texts)

#print (docs[0])
#print(docs[1])


#POS TAGGING
print("POS TAGGING--------------------------------------------------------------------------------------------------")

data = []
for i, doc in enumerate(docs):

    # Obtain the list of tokens in the document
    tokens = [t for t, label in doc]

    # Perform POS tagging
    tagged = nltk.pos_tag(tokens)

    # Take the word, POS tag, and its label
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
    
def word2features(doc, i):
    word = doc[i][0]
    postag = doc[i][1]

    # Common features for all words
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag
    ]

    # Features for words that are not
    # at the beginning of a document
    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:word.isdigit=%s' % word1.isdigit(),
            '-1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'beginning of a document'
        features.append('BOS')

    # Features for words that are not
    # at the end of a document
    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:word.isdigit=%s' % word1.isdigit(),
            '+1:postag=' + postag1
        ])
    else:
        # Indicate that it is the 'end of a document'
        features.append('EOS')

    return features



print("TESTING AND TRAINING DATA PREPARATION --------------------------------------------------------------------------------------------------")

# A function for extracting features in documents
def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]

# A function fo generating the list of labels for each document
def get_labels(doc):
    return [label for (token, postag, label) in doc]

X = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

#print(X_train[0])
#print("-------------------------------------------")
#print(X_test[0])

print("Training --------------------------------------------------------------------------------------------------")


trainer = pycrfsuite.Trainer(verbose=True)

# Submit training data to the trainer
for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)

# Set the parameters of the model
trainer.set_params({
    # coefficient for L1 penalty
    'c1': 0.01,

    # coefficient for L2 penalty
    'c2': 0.1,  

    # maximum number of iterations
    'max_iterations': 100,

    # whether to include transitions that
    # are possible, but not observed
    'feature.possible_transitions': True
})

# Provide a file name as a parameter to the train function, such that
# the model will be saved to the file when training is finished
trainer.train('crf.model')


print("Testing --------------------------------------------------------------------------------------------------")

tagger = pycrfsuite.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in X_test]

# Let's take a look at a random sample in the testing set
i = 15
for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in X_test[i]]):
    print("%s (%s)" % (y, x))


print("PERFORMANCE MEASURE --------------------------------------------------------------------------------------------------")

# Create a mapping of labels to indices
labels = {"O": 5, "U": 4, "P": 3, "N": 2, "C": 1, "S": 0}

# Convert the sequences of tags into a 1-dimensional array
predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])

# Print out the classification report
print(classification_report(
    truths, predictions,
    target_names=["S", "C", "N", "P", "U", "O"]))
