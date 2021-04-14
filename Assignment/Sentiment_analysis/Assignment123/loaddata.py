
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
import sys
import os
import string, nltk
#nltk.download('stopwords')
nltk.data.path.append("/home/ubuntu/nltk_data")
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
#nltk.download('punkt')
#nltk.download('wordnet')
import numpy as np
import torch
from string import punctuation
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from torch import nn, optim
from torchinfo import summary
import pandas as pd
import tensorflow as tf
import re
import matplotlib.pyplot as plt

wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english') + list(string.punctuation)) 
device="cuda"
def create_embedding_matrix(word_index,embedding_dict,dimension):
    embedding_matrix=np.zeros((len(word_index)+1,dimension))

    for word,index in word_index.items():
        if word in embedding_dict:
            embedding_matrix[index]=embedding_dict[word]
    return embedding_matrix


def Norm(text,wordnet_lemmatizer,stop_words):
    text = text.lower().strip()
    text =  re.sub(' +', ' ', text)
    word_tokens = word_tokenize(text) 
    filtered_sentence = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            w = wordnet_lemmatizer.lemmatize(w, pos="v")
            filtered_sentence.append(w) 
    texts=" ".join(str(x) for x in filtered_sentence)
    return text

def pad_text(encoded_reviews, seq_length):
    
    reviews = []
    
    for review in encoded_reviews:
        print(len(review))
        if len(review) >= seq_length:
            reviews.append(review[:seq_length])
        else:
            reviews.append([0]*(seq_length-len(review)) + review)
        
    return np.array(reviews)

def LoadData(file, Vocab):
    with open(file, "r",encoding="ISO-8859-1") as f:
        data_x = []
        data_y = []
        contents = f.read().splitlines()
        for line in contents:
            try:
                _,text,label = line.split("#")
            except:
                continue
            text = text.split(" ",1)[1]
            
            text = "".join([ch for ch in text if ch not in punctuation])
            text = Norm(text,wordnet_lemmatizer,stop_words)
            
            data_x.append(text)
            data_y.append(label)
            Vocab = Vocab + text.split(" ")
        return data_x, data_y, Vocab


def GetData():
    seqence_len = 150
    embed_len = 300
    batch_size = 50
    Vocab = []

    glove = pd.read_csv("../../wordEmbedding/"+'glove.6B.'+str(embed_len)+'d.txt', sep=" ", quoting=3, header=None, index_col=0)
    glove_embedding = {key: val.values for key, val in glove.T.items()}

    train_x, train_y, Vocab = LoadData("../data/train.txt",Vocab)
    dev_x, dev_y, Vocab = LoadData("../data/dev.txt",Vocab)
    test_x, test_y, Vocab = LoadData("../data/test.txt",Vocab)

    tokenizer=tf.keras.preprocessing.text.Tokenizer(split=" ")
    tokenizer.fit_on_texts(train_x+dev_x+test_x)

    encoded_train =tokenizer.texts_to_sequences(train_x)
    encoded_dev =tokenizer.texts_to_sequences(dev_x)
    encoded_test =tokenizer.texts_to_sequences(test_x)


    train_x = pad_text(encoded_train, seq_length = seqence_len)
    train_y = np.array([1 if label == "pos" else 0 for label in train_y])


    dev_x = pad_text(encoded_dev, seq_length = seqence_len)
    dev_y = np.array([1 if label == "pos" else 0 for label in dev_y])


    test_x = pad_text(encoded_test, seq_length = seqence_len)
    test_y = np.array([1 if label == "pos" else 0 for label in test_y])

    # print(len(type(encoded_test)))


    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(dev_x), torch.from_numpy(dev_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

    embedding_matrix=create_embedding_matrix(tokenizer.word_index,embedding_dict=glove_embedding,dimension=embed_len)

    return train_loader, valid_loader, test_loader, embedding_matrix