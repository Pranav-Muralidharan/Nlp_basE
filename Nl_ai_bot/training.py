from nltk.sem.evaluate import Model
from nltk.tag.brill import Word
import numpy as np
from tensorflow.keras import layers, models, optimizers

import random
import json 
import pickle

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intent.json').read())

words = []
classes = []
documents = []
ignorents = ['.', ',', '?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_lst = nltk.word_tokenize(pattern)
        words.extend(word_lst)
        documents.append((word_lst, intent['Tag']))
        if intent['Tag'] not in classes:
            classes.append(intent['Tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignorents]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_pattern = document[0]
    word_pattern = [lemmatizer.lemmatize(word.lower()) for word in word_pattern]
    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_X = list(training[:, 0])
train_y = list(training[:, 1])

model = models.Sequential()
model.add(layers.Dense(128, input_shape=(len(train_X[0]),), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(train_y[0]), activation='softmax'))

sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=1e-6, nesterov=True)

model.compile(optimizer='sgd',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

hist = model.fit(np.array(train_X), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('AI_bot.h5', hist)
print('done')

