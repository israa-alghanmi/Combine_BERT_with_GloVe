# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:19:11 2020

@author: Israa
"""


import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from transformers import *
from tensorflow.keras import layers

################ BERT+Glove#####################pooled layer###################



def CNN_BERT_Glove(model_hf2, embedding_matrix,seed_value, num_classes):
        
    input_ids_in = tf.keras.layers.Input(shape=(250,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(250,), name='masked_token', dtype='int32') 
    embedding_layer = model_hf2([input_ids_in,input_masks_in])[1][12]
    CNN_input = tf.keras.layers.Input(shape=(250,), name='CNN_input') 
    embedding_layer2 = layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=250 , trainable=False)(CNN_input)
    

    query_seq_encoding = tf.keras.layers.Conv1D(filters=100,kernel_size=3,padding='same')(embedding_layer2)
    query_seq_encoding = tf.keras.layers.GlobalMaxPooling1D()(query_seq_encoding)
    query_seq_encoding2 = tf.keras.layers.GlobalAveragePooling1D()(embedding_layer)
    input_layer = tf.keras.layers.Concatenate()([query_seq_encoding, query_seq_encoding2])
    x = layers.Dropout(0.5, seed=seed_value)(input_layer)
    o = layers.Dense(num_classes, activation='softmax')(x)


    
    model_cnn2 = tf.keras.Model(inputs=[input_ids_in, input_masks_in, CNN_input], outputs=o)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()    
    optimizer='SGD'

    metric = tf.keras.metrics.SparseCategoricalAccuracy('sparse_categorical_accuracy')
    model_hf2.trainable = False
    model_cnn2.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model_cnn2


def CNN_BERT_Glovetwitt(model_hf2, embedding_matrix,seed_value,num_classes, train_inputs_v01, valid_inputs_v01, test_inputs_v01, train_masks_v01, valid_masks_v01, test_masks_v01,train_data, valid_data, test_data ,y_train, y_valid, y_test):
    
    my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)]
    
    tf.compat.v1.reset_default_graph()
    cnnmodel = CNN_BERT_Glove(model_hf2,embedding_matrix, seed_value, num_classes)
    cnnmodel.fit([train_inputs_v01, train_masks_v01, train_data], y_train, validation_data =([valid_inputs_v01, valid_masks_v01,valid_data],y_valid), epochs=15,verbose=0, batch_size=16,callbacks=my_callbacks)
    predictions =  cnnmodel.predict([test_inputs_v01, test_masks_v01,test_data])
    yhat = np.argmax(predictions, axis=1)
    acc2 = accuracy_score(y_test, yhat)
    f1=f1_score(y_test, yhat,average='macro' )
    print('CNN(BERT+Glove)  Accuracy: %.5f' % acc2)
    print('CNN(BERT+Glove)  F1: %.5f' % f1)
        




def lstm_model1(model_hf2, embedding_matrix,seed_value,num_classes):
        
    input_ids_in = tf.keras.layers.Input(shape=(250,), name='input_token', dtype='int32')
    input_masks_in = tf.keras.layers.Input(shape=(250,), name='masked_token', dtype='int32') 
    embedding_layer = model_hf2([input_ids_in,input_masks_in])[1][12]
    CNN_input = tf.keras.layers.Input(shape=(250,), name='CNN_input') 
    embedding_layer2 = layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=250 , trainable=False)(CNN_input)
    

    query_seq_encoding = tf.keras.layers.LSTM(units=100)(embedding_layer2)
    query_seq_encoding2 = tf.keras.layers.GlobalAveragePooling1D()(embedding_layer)
    input_layer = tf.keras.layers.Concatenate()([query_seq_encoding, query_seq_encoding2])
    x = layers.Dropout(0.5, seed=seed_value)(input_layer)
    o = layers.Dense(num_classes, activation='softmax')(x)

    
    model_lstm = tf.keras.Model(inputs=[input_ids_in, input_masks_in, CNN_input], outputs=o)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()    
    optimizer='SGD'

    metric = tf.keras.metrics.SparseCategoricalAccuracy('sparse_categorical_accuracy')
    model_hf2.trainable = False
    model_lstm.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    return model_lstm




def LSTM_BERT_Glovetwitt(model_hf2, embedding_matrix, seed_value,num_classes, train_inputs_v01, valid_inputs_v01, test_inputs_v01, train_masks_v01, valid_masks_v01, test_masks_v01,train_data, valid_data, test_data ,y_train, y_valid, y_test):

    my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)]
    tf.compat.v1.reset_default_graph()
    lstm_model = lstm_model1(model_hf2,embedding_matrix, seed_value, num_classes)
    lstm_model.fit([train_inputs_v01, train_masks_v01, train_data], y_train, validation_data =([valid_inputs_v01, valid_masks_v01,valid_data],y_valid), epochs=15,verbose=0, batch_size=16,callbacks=my_callbacks)
    predictions =  lstm_model.predict([test_inputs_v01, test_masks_v01,test_data])
    yhat = np.argmax(predictions, axis=1)
    acc2 = accuracy_score(y_test, yhat)
    f1=f1_score(y_test, yhat,average='macro' )
    print('LSTM(BERT+Glove)  Accuracy: %.5f' % acc2)
    print('LSTM(BERT+Glove) F1: %.5f' % f1)


        










    
    
