from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import random
import json
import os

#load dataset
with open(os.path.join(os.path.dirname(__file__), 'BOT_CONFIG - 22.06.2021.json')) as f:
    BOT_CONFIG = json.load(f)

def bot(word, classifier, vectorizer):
    #bot's communications
    return random.choice(BOT_CONFIG['intents'][classifier.predict(vectorizer.transform([word]))[0]]['responses'])

def prepare_data():
    #prepare dataset for learning model
    X, y = [], []
    for intent in BOT_CONFIG['intents']:
        for example in BOT_CONFIG['intents'][intent]['examples']:
            X.append(example)
            y.append(intent)
    return X, y

def learn_model():
    X, y = prepare_data()
    #convert words into nums
    vectorizer = TfidfVectorizer(analyzer='char')
    vectorizer.fit(X)
    X_vectors = vectorizer.transform(X)
    #learn model
    rf = RandomForestClassifier(n_estimators=100,criterion='gini')
    rf.fit(X_vectors,y)
    return X_vectors, y, rf, vectorizer

if __name__ == '__main__':
    X, y, rf, vectorizer = learn_model()
    print('print "exit" to quit from chat, model accuracy is {0}%'.format(round(rf.score(X,y)*100,2)))
    #main cycle
    word = ''
    while True:
        word = str(input())
        if word == 'exit':break
        print(bot(word,rf,vectorizer))
