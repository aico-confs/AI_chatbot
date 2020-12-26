import random
import json
import pickle
import numpy as np

import nltk
import patterns as patterns
from nltk.stem import WordNetLemmatizer
import jieba
# nltk.download()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, AbstractRNNCell, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
jieba.set_dictionary('dict.txt.big.txt')

intents = json.loads(open(r"C:\Users\q1233\OneDrive\Desktop\程式自學\python\chatbot\intents.json",mode="r",encoding="UTF-8").read())

words = []
classes = []
documents = []
ignore_letters = ["!", '，', '。']
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # word_list = nltk.word_tokenize(pattern)
        word_list = list(jieba.cut(pattern, cut_all=False,HMM=True))
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# # print(documents)
# [(['你好'], 'greetings'), (['嗨'], 'greetings'), (['哈', '囉'], 'greetings'), (['欸'], 'greetings'), (['诶'], 'greetings'), (['再見'], 'bye'), (['掰'], 'bye'), (['拜'], 'bye'), (['掰掰'],
#  'bye'), (['拜拜'], 'bye'), (['明天', '見'], 'bye')]

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# print(words)
words = sorted(set(words))
# # print(words)
# ['你好', '嗨', '哈', '囉', '欸', '诶', '再見', '掰', '拜', '掰掰', '拜拜', '明天', '見']
# ['你好', '再見', '哈', '嗨', '囉', '拜', '拜拜', '掰', '掰掰', '明天', '欸', '見', '诶']



classes = sorted(set(classes))
with open('words.pkl', 'wb') as file:
    pickle.dump(words, file)
with open('classes.pkl', 'wb') as file:
    pickle.dump(classes, file)

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
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='relu'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

print('yeah')



