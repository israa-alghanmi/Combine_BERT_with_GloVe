# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:54:02 2020

@author: Israa
"""


import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import TreebankWordTokenizer
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from transformers import *


def toknize_text(data_train, data_valid, data_test):
    new_tokenizer = Tokenizer()
    new_tokenizer.fit_on_texts(data_train+data_valid+data_test)
    data_train= new_tokenizer.texts_to_sequences(data_train)
    data_train= pad_sequences(data_train, padding='post', maxlen=250)
    data_valid= new_tokenizer.texts_to_sequences(data_valid)
    data_valid= pad_sequences(data_valid, padding='post', maxlen=250)
    data_test= new_tokenizer.texts_to_sequences(data_test)
    data_test= pad_sequences(data_test, padding='post', maxlen=250)
    vocab_size = len(new_tokenizer.word_index) + 1
    return data_train,data_valid,data_test, vocab_size,new_tokenizer


def get_final_data(xTrain,xValid,xTest):
    data_train=[]
    data_valid=[]
    data_test=[]
    for sentence in list(xTrain):
        sentence= TreebankWordTokenizer().tokenize(sentence)
        data_train.append(sentence)
    for sentence in list(xValid):
        sentence= TreebankWordTokenizer().tokenize(sentence)
        data_valid.append(sentence)
    for sentence in list(xTest):
        sentence= TreebankWordTokenizer().tokenize(sentence)
        data_test.append(sentence)

    train_data, valid_data, test_data, n_vocab, n_tokenizer= toknize_text(data_train, data_valid,data_test)
    train_data=np.array(train_data)
    valid_data=np.array(valid_data)
    test_data=np.array(test_data)
    return train_data, valid_data, test_data, n_vocab, n_tokenizer



def getEmbeddingMatrix(n_vocab, n_tokenizer,pretrainedEmbedding):
    i=0
    embedding_matrix= zeros((n_vocab, 100))
    oov_words=[]
    for word , index in n_tokenizer.word_index.items():
        embedding_vector=None
        try:
            if word in pretrainedEmbedding.wv:
                embedding_matrix[index] = pretrainedEmbedding[word]
            else: 
                i+=1
                oov_words.append(word)
        except:
            continue
    return embedding_matrix, oov_words,i


def get_glove_toknized_text(df_train,df_valid, df_test, golve_twitter):
    xTrain=df_train['tweet']
    y_train=df_train.label.values
    
    xValid=df_valid['tweet']
    y_valid=df_valid.label.values
    
    xTest=df_test['tweet']
    y_test=df_test.label.values
    
    train_data, valid_data, test_data, n_vocab, n_tokenizer= get_final_data(xTrain,xValid,xTest)
    embedding_matrix, oov_words,count_oov=getEmbeddingMatrix(n_vocab, n_tokenizer, golve_twitter)
    
    return train_data, valid_data, test_data, n_vocab, n_tokenizer, embedding_matrix, oov_words,count_oov,y_train, y_valid, y_test

def prepareBertInput(tokenizer,sentences):
        attention_mask=[]
        input_ids=[]
        tokenized = sentences.apply((lambda x: tokenizer.encode(str(x), add_special_tokens=True)))
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)
        for sentence in sentences:
            tokenized2=tokenizer.encode_plus(str(sentence),  max_length=250, pad_to_max_length=True,add_special_tokens=True)
            attention_mask.append(tokenized2['attention_mask'])
            input_ids.append(tokenized2['input_ids'])
    
        return input_ids , attention_mask, max_len,tokenized 

def get_bert_inputs(df_train,df_valid, df_test):
    ##BERT 
    bert_tokenizer_transformer_v01 = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    xTrain=df_train['tweet']
    xValid=df_valid['tweet']
    xTest=df_test['tweet']
    
    train_inputs_v01, train_masks_v01, max_len_v01,tokenized =prepareBertInput(bert_tokenizer_transformer_v01,xTrain )
    valid_inputs_v01, valid_masks_v01, max_len_v01_valid,tokenized_valid =prepareBertInput(bert_tokenizer_transformer_v01,xValid )
    test_inputs_v01, test_masks_v01, max_len_v01_test,tokenized_test =prepareBertInput(bert_tokenizer_transformer_v01,xTest )
    
    
    train_inputs_v01=tf.constant(train_inputs_v01)
    valid_inputs_v01=tf.constant(valid_inputs_v01)
    test_inputs_v01=tf.constant(test_inputs_v01)
    train_masks_v01=tf.constant(train_masks_v01)
    valid_masks_v01=tf.constant(valid_masks_v01)
    test_masks_v01=tf.constant(test_masks_v01)
    
    return train_inputs_v01, valid_inputs_v01, test_inputs_v01, train_masks_v01, valid_masks_v01, test_masks_v01