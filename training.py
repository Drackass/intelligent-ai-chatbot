import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))

    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# print(documents)
# [(['Hello'], 'greeting'), (['Hi'], 'greeting'), (['Hey'], 'greeting'), (['Good', 'morning'], 'greeting'), (['Good', 'afternoon'], 'greeting'), (['Good', 'evening'], 'greeting'), (['Bye'], 'goodbye'), (['Goodbye'], 'goodbye'), (['See', 'you', 'later'], 'goodbye'), (['Good', 'night'], 'goodbye'), (['Thank', 'you'], 'thanks'), (['Thanks'], 'thanks'), (['Thank', 'you', 'very', 'much'], 'thanks'), (['Help'], 'help'), (['Can', 'you', 'help', 'me', '?'], 'help'), (['I', 'need', 'help'], 'help'), (['What', "'s", 'the', 'weather', 'like', '?'], 'weather'), (['How', "'s", 'the', 'weather', '?'], 'weather'), (['Is', 'it', 'raining', '?'], 'weather')]

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# print(words)
# ['hello', 'hi', 'hey', 'good', 'morning', 'afternoon', 'evening', 'bye', 'goodbye', 'see', 'you', 'later', 'thank', 'you', 'thanks', 'help', 'can', 'you', 'me', 'what', "'s", 'the', 'weather', 'like', 'how', "'s", 'raining']

classes = sorted(set(classes))

# print(classes)
# ['greeting', 'goodbye', 'thanks', 'help', 'weather']

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.keras', hist)
print("Model created and saved")
