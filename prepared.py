import random
import json
import pickle
import numpy as np

import jieba
import jieba.posseg as pseg

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

jieba.set_dictionary('dict.txt.big.txt')
lemmatizer = WordNetLemmatizer()

with open(r"C:\Users\q1233\OneDrive\Desktop\程式自學\python\chatbot\intents.json", mode="r", encoding="UTF-8") as file:
    intents = json.loads(file.read())
with open('words.pkl', 'rb') as file:
    words = pickle.load(file)
with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    # sentence_words = nltk.word_tokenize(sentence)
    sentence_words = jieba.cut(sentence, cut_all=False)

    sentence_words = list(sentence_words)
    # sentence_words = list(pseg.cut(sentence))
    # for word, flag in words:
    #     print((word, flag))
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    print(sentence_words)
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    print(np.array(bag))
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    print(res)
    ERROR_THRESHOLD = 25
    # results = [[i, r] for i, r in enumerate(res) if i > ERROR_THRESHOLD]
    results = [[i, r] for i, r in enumerate(res)]

    # results = [[i, r] for i, r in enumerate(res) ]
    print(results)
    results.sort(key=lambda x: x[1], reverse=True)
    print(results)
    results_list = []
    for r in results:
        results_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return results_list
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
print("start")
while True:

    message = input("")
    ints = predict_class(message)

    res = get_response(ints, intents)
    print(res)
