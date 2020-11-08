# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:59:24 2020

@author: Israa
"""

from preprocessData import *
from BERTandGlove import *
from tokenization import *

import sys
import argparse

import os
import random
import numpy as np
import tensorflow as tf
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from sklearn.metrics import accuracy_score, f1_score

from transformers import *

from tensorflow.compat.v1.keras.backend import set_session
#############





def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Experiment setup - Arguments get parsed via --commands')

    parser.add_argument('--traint', dest='training_path_text', type=str, default='./Dataset/irony/train_text.txt',
        help='Training data - text path')
    parser.add_argument('--trainl', dest='training_path_labels', type=str, default='./Dataset/irony/train_labels.txt',
        help='Training data - labels path')
    parser.add_argument('--validt', dest='validation_path_text', type=str, default='./Dataset/irony/val_text.txt',
        help='Validation data - text path')
    parser.add_argument('--validl', dest='validation_path_labels', type=str, default='./Dataset/irony/val_labels.txt',
        help='Validation data - labels path')
    parser.add_argument('--textt', dest='testing_path_text', type=str, default='./Dataset/irony/test_text.txt',
        help='testing data - text path')
    parser.add_argument('--testl', dest='testing_path_labels', type=str, default='./Dataset/irony/test_labels.txt',
        help='testing data - labels path')
    parser.add_argument('--glovepath', dest='glove_file_path', type=str, default='/EnglishTwitterGlove/glove.twitter.27B.100d.txt',
        help='Glove File path')
    parser.add_argument('--finetunedbert', dest='finetuned_bert_path', type=str, default='./Fine-tuned_BERT', 
        help='testing data - labels path')
    args = parser.parse_args()
    '''
     parser.add_argument('--v', metavar='verbosity', type=int, default=2,
        help='Verbosity of logging: 0 -critical, 1- error, 2 -warning, 3 -info, 4 -debug')
    args = vars(parser.parse_args())
    handler = args.pop('handler')
    handler(**args)
    verbose = {0: logging.CRITICAL, 1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}
    logging.basicConfig(format='%(message)s', level=verbose[args.v], stream=sys.stdout)

    '''

    return args
    
def set_seed(seed_value):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)
    # 5. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

def load_Glove(gloveFile):

    glove_file = datapath( os.getcwd()+gloveFile)
    tmp_file = get_tmpfile( os.getcwd()+'/EnglishTwitterGlove/test_word2vec.txt')
    _ = glove2word2vec(glove_file, tmp_file)
    golve_twitter = KeyedVectors.load_word2vec_format(tmp_file)
    return golve_twitter

def load_fine_tuned_bert(finetuned_bert_path,test_inputs_v01,test_masks_v01,y_test):
    ########Load the separately fine-tuned BERT checkpoint#########
    model_hf2 = TFBertForSequenceClassification.from_pretrained(finetuned_bert_path)
    print('Printing ...........')
    ########uncomment to print results#########
    '''
    predictions_bert = model_hf2.predict([test_inputs_v01, test_masks_v01])
    yhat_bert = np.argmax(predictions_bert[0], axis=1)
    acc = accuracy_score(y_test, yhat_bert)
    f1=f1_score(y_test, yhat_bert,average='macro' )
    print('Fine-tuned BERT Test Accuracy: %.5f' % acc)
    print('Fine-tuned BERT Test F1: %.5f' % f1)
    '''
    
    return model_hf2
    
def main():
    args = parse_arguments()
    seed_value= 12321
    set_seed(seed_value)
    df_train, df_valid, df_test= get_preprocessed_data(args.training_path_text,args.training_path_labels,args.validation_path_text,args.validation_path_labels,args.testing_path_text,args.testing_path_labels)
    golve_twitter= load_Glove(args.glove_file_path)
    num_classes= len(set(df_train['label']))
    train_data, valid_data, test_data, n_vocab, n_tokenizer, embedding_matrix, oov_words,count_oov,y_train, y_valid, y_test= get_glove_toknized_text(df_train, df_valid, df_test,golve_twitter)
    train_inputs_v01, valid_inputs_v01, test_inputs_v01, train_masks_v01, valid_masks_v01, test_masks_v01= get_bert_inputs(df_train, df_valid, df_test)
    model_hf2= load_fine_tuned_bert(args.finetuned_bert_path,test_inputs_v01,test_masks_v01,y_test)
    CNN_BERT_Glovetwitt(model_hf2, embedding_matrix,seed_value, num_classes, train_inputs_v01, valid_inputs_v01, test_inputs_v01, train_masks_v01, valid_masks_v01, test_masks_v01,train_data, valid_data, test_data ,y_train, y_valid, y_test)
    LSTM_BERT_Glovetwitt(model_hf2, embedding_matrix,seed_value, num_classes, train_inputs_v01, valid_inputs_v01, test_inputs_v01, train_masks_v01, valid_masks_v01, test_masks_v01,train_data, valid_data, test_data ,y_train, y_valid, y_test)
    
    
    
    
if __name__ == '__main__':  
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    main()
    print('Done!')