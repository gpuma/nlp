from sklearn.naive_bayes import MultinomialNB
#required for word frequency count
from scipy.stats import itemfreq
import pandas as pd
import numpy as np
import string

data = pd.read_csv('spambase.data').as_matrix()
np.random.shuffle(data)

#first 48 columns is the frequency of words
X = data[:, :48]

#print X[1]
#print type(X[1])

#last column indicates if it's spam
Y = data[:, -1]

#first n-100 rows
Xtrain = X[:-100,]
Ytrain = Y[:-100,]

#last 100 rows
Xtest = X[-100:,]
Ytest = Y[-100:,]

model = MultinomialNB()
model.fit(Xtrain, Ytrain)

print "Classification rate for NB", model.score(Xtest, Ytest)

from sklearn.ensemble import AdaBoostClassifier
model2 = AdaBoostClassifier()
model2.fit(Xtrain, Ytrain)

#print "Classification rate for AdaBoost", model2.score(Xtest, Ytest)

print 'reading sample file'
#pre-processing
with open('spamwords.txt', 'r') as myfile:
	words = myfile.read().split()

print 'preprocessing file'
#table for replacing punctuation with spaces
replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))

#feature extraction
with open('test2.txt','r') as myfile:
	#splitlines() unlike readline() removes the \n character
	#translate() uses the table replace_punctuation from above
	#we are interested in words only, all lowercase
	data = myfile.read().translate(replace_punctuation).lower().split()

print 'extracting features'
#number of words in the email
n=len(data)
#we are only interested in the frequency 48 words (features) described in the documentation (spambase.names)
data = [d for d in data if d in words]
#we change our frequency table to a dictionary for easy access later
freq = dict(itemfreq(data))

#dict.get() allows us to specify a default value (0) in case the key doesn't exist
full_frequency = [[w, 100 * int(freq.get(w,0)) / n] for w in words]
#we use astype cause it's originally an array of strings
features = np.array(full_frequency)[:,1].astype(np.float)

print full_frequency

print features

#features = [0,0,0,0,0,0,0,0,0,0,0,1.12,0,0,0,0,0,0,2.24,0,0.56,0,0,0.56,0,0,1.12,0,0,0,0,0,0,0,0,0,0.56,0,0,0.56,0,0,0.56,0,0.56,0,0,0]

print 'prediction according to NB: ', model2.predict(features)
