
from logging import ERROR
import random
import json 
import pickle
from google.protobuf import message

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intent.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('AI_bot.h5')

def filterd_sentence(sentence):
    sent_word = nltk.word_tokenize(sentence)
    sent_word = [lemmatizer.lemmatize(word) for word in sent_word]
    return sent_word

def word_bag(sentence):
    sent_word = filterd_sentence(sentence)
    bag = [0] * len(words)
    for j in sent_word:
        for i, word in enumerate(words):
            if word == j:
                bag[i] = 1
    return np.array(bag)

def classpredictor(sentence):
    ba = word_bag(sentence)
    pd = model.predict(np.array([ba]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r]for i, r in enumerate(pd) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1], reverse=True)
    rs = []
    for r in results:
        rs.append({'intent':classes[r[0]], 'probability':str([1])})
    return rs

def response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['Tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('haa!...mm working :)')

while True:
    message = input('')
    ints = classpredictor(message)
    resp = response(ints, intents)
    print(resp)

