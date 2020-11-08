# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:09:54 2020

@author: Israa
"""

import pandas as pd
import re
import emoji

    
def load_dict_smileys():

    return {
        ":‑)":"smiley",
        ":-]":"smiley",
        ":-3":"smiley",
        ":->":"smiley",
        "8-)":"smiley",
        ":-}":"smiley",
        ":)":"smiley",
        ":]":"smiley",
        ":3":"smiley",
        ":>":"smiley",
        "8)":"smiley",
        ":}":"smiley",
        ":o)":"smiley",
        ":c)":"smiley",
        ":^)":"smiley",
        "=]":"smiley",
        "=)":"smiley",
        ":-))":"smiley",
        ":‑D":"smiley",
        "8‑D":"smiley",
        "x‑D":"smiley",
        "X‑D":"smiley",
        ":D":"smiley",
        "8D":"smiley",
        "xD":"smiley",
        "XD":"smiley",
        ":‑(":"sad",
        ":‑c":"sad",
        ":‑<":"sad",
        ":‑[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'‑(":"sad",
        ":'(":"sad",
        ":‑P":"playful",
        "X‑P":"playful",
        "x‑p":"playful",
        ":‑p":"playful",
        ":‑Þ":"playful",
        ":‑þ":"playful",
        ":‑b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":Þ":"playful",
        ":þ":"playful",
        ":b":"playful",
        "<3":"love"
        }

# self defined contractions
def load_dict_contractions():
    
    return {
        "ain't":"is not",
        "amn't":"am not",
        "aren't":"are not",
        "can't":"cannot",
        "'cause":"because",
        "couldn't":"could not",
        "couldn't've":"could not have",
        "could've":"could have",
        "daren't":"dare not",
        "daresn't":"dare not",
        "dasn't":"dare not",
        "didn't":"did not",
        "doesn't":"does not",
        "don't":"do not",
        "e'er":"ever",
        "em":"them",
        "everyone's":"everyone is",
        "finna":"fixing to",
        "gimme":"give me",
        "gonna":"going to",
        "gon't":"go not",
        "gotta":"got to",
        "hadn't":"had not",
        "hasn't":"has not",
        "haven't":"have not",
        "he'd":"he would",
        "he'll":"he will",
        "he's":"he is",
        "he've":"he have",
        "how'd":"how would",
        "how'll":"how will",
        "how're":"how are",
        "how's":"how is",
        "I'd":"I would",
        "I'll":"I will",
        "I'm":"I am",
        "I'm'a":"I am about to",
        "I'm'o":"I am going to",
        "isn't":"is not",
        "it'd":"it would",
        "it'll":"it will",
        "it's":"it is",
        "I've":"I have",
        "kinda":"kind of",
        "let's":"let us",
        "mayn't":"may not",
        "may've":"may have",
        "mightn't":"might not",
        "might've":"might have",
        "mustn't":"must not",
        "mustn't've":"must not have",
        "must've":"must have",
        "needn't":"need not",
        "ne'er":"never",
        "o'":"of",
        "o'er":"over",
        "ol'":"old",
        "oughtn't":"ought not",
        "shalln't":"shall not",
        "shan't":"shall not",
        "she'd":"she would",
        "she'll":"she will",
        "she's":"she is",
        "shouldn't":"should not",
        "shouldn't've":"should not have",
        "should've":"should have",
        "somebody's":"somebody is",
        "someone's":"someone is",
        "something's":"something is",
        "that'd":"that would",
        "that'll":"that will",
        "that're":"that are",
        "that's":"that is",
        "there'd":"there would",
        "there'll":"there will",
        "there're":"there are",
        "there's":"there is",
        "these're":"these are",
        "they'd":"they would",
        "they'll":"they will",
        "they're":"they are",
        "they've":"they have",
        "this's":"this is",
        "those're":"those are",
        "'tis":"it is",
        "'twas":"it was",
        "wanna":"want to",
        "wasn't":"was not",
        "we'd":"we would",
        "we'd've":"we would have",
        "we'll":"we will",
        "we're":"we are",
        "weren't":"were not",
        "we've":"we have",
        "what'd":"what did",
        "what'll":"what will",
        "what're":"what are",
        "what's":"what is",
        "what've":"what have",
        "when's":"when is",
        "where'd":"where did",
        "where're":"where are",
        "where's":"where is",
        "where've":"where have",
        "which's":"which is",
        "who'd":"who would",
        "who'd've":"who would have",
        "who'll":"who will",
        "who're":"who are",
        "who's":"who is",
        "who've":"who have",
        "why'd":"why did",
        "why're":"why are",
        "why's":"why is",
        "won't":"will not",
        "wouldn't":"would not",
        "would've":"would have",
        "y'all":"you all",
        "you'd":"you would",
        "you'll":"you will",
        "you're":"you are",
        "you've":"you have",
        "Whatcha":"What are you",
        "luv":"love",
        "sux":"sucks"
        }




def remove_redundant_punct(text,redundant_punct_pattern):
    text_ = text
    result = re.search(redundant_punct_pattern, text)
    dif = 0
    while result:
        sub = result.group()
        sub = sorted(set(sub), key=sub.index)
        sub = ' ' + ''.join(list(sub)) + ' '
        text = ''.join((text[:result.span()[0]+dif], sub, text[result.span()[1]+dif:]))
        text_ = ''.join((text_[:result.span()[0]], text_[result.span()[1]:])).strip()
        dif = abs(len(text) - len(text_))
        result = re.search(redundant_punct_pattern, text_)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess(text):
    regex_url_step1 = r'(?=http)[^\s]+'
    regex_url_step2 = r'(?=www)[^\s]+'
    regex_url = r'(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
    regex_mention = r'@[\w\d]+'
    regex_email = r'\S+@\S+'
    redundant_punct_pattern = r'([!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ【»؛\s+«–…‘]{2,})'

    text=str(text)
    processing_tweet = re.sub('ـ', '', text)
    processing_tweet= processing_tweet.lower()
    processing_tweet = re.sub('[«»]', ' " ', processing_tweet)
    processing_tweet = re.sub(regex_url_step1, '[link]', processing_tweet)
    processing_tweet = re.sub(regex_url_step2, '[link]', processing_tweet)
    processing_tweet = re.sub(regex_url, '[link]', processing_tweet)
    processing_tweet = re.sub(regex_email, '[email]', processing_tweet)
    processing_tweet = re.sub(regex_mention, '[user]', processing_tweet)
    processing_tweet = re.sub('…', r'\.', processing_tweet).strip()
    processing_tweet = remove_redundant_punct(processing_tweet, redundant_punct_pattern)
    processing_tweet = re.sub(r'\[ link \]|\[ link\]|\[link \]', ' [link] ', processing_tweet)
    processing_tweet = re.sub(r'\[ email \]|\[ email\]|\[email \]', ' [email] ', processing_tweet)
    processing_tweet = re.sub(r'\[ user \]|\[ user\]|\[user \]', ' [user] ', processing_tweet)
    processing_tweet = re.sub("(.)\\1{2,}", "\\1", processing_tweet)
    processing_tweet=strip_emoji(processing_tweet)

    search = ['_','\\','\n','-', ',','/' ,'.','\t','?','!','+','*','\'','|','#', '$','%']
    replace = [' ', ' ',' ',' ', ' ',' ', ' ',' ',' ',' ',' ',' ',' ',' ', ' ', ' ',' ']
    #remove numbers
    processing_tweet = re.sub(r'\d+', '', processing_tweet)
    processing_tweet = ' '.join(re.sub("[\n\.\,\"\!\?\:\;\-\=\؟]", " ", processing_tweet).split())
    processing_tweet = ' '.join(re.sub("[\_]", " ", processing_tweet).split())
    processing_tweet = re.sub(r'[^\x00-\x7F]+',' ', processing_tweet)

    for i in range(0, len(search)):
        processing_tweet = processing_tweet.replace(search[i], replace[i])

    return processing_tweet.strip()

def strip_emoji(text):
    new_text = re.sub(emoji.get_emoji_regexp(), r" ", text)
    return new_text

def get_preprocessed_data(training_path_text,training_path_labels,validation_path_text,validation_path_labels,testing_path_text,testing_path_labels):
    
    #Train
    data_file_train_text =open(training_path_text, "r", encoding='utf-8')
    datatable_arabic_train_text = [preprocess(str(line)) for line in data_file_train_text.read().splitlines()]
    data_file_train_labels =open(training_path_labels, "r", encoding='utf-8')
    datatable_arabic_train_labels = [int(line) for line in data_file_train_labels.read().splitlines()]
    d = {'tweet': datatable_arabic_train_text, 'label': datatable_arabic_train_labels}
    df_train = pd.DataFrame(data=d)
    
    
    #Val
    data_file_valid_text =open(validation_path_text, "r", encoding='utf-8')
    datatable_arabic_valid_text = [preprocess(str(line)) for line in data_file_valid_text.read().splitlines()]
    data_file_valid_labels =open(validation_path_labels, "r", encoding='utf-8')
    datatable_arabic_valid_labels = [int(line) for line in data_file_valid_labels.read().splitlines()]
    d2 = {'tweet': datatable_arabic_valid_text, 'label': datatable_arabic_valid_labels}
    df_valid = pd.DataFrame(data=d2)
    
    
    #Test
    data_file_test_text =open(testing_path_text, "r", encoding='utf-8')
    datatable_arabic_test_text = [preprocess(str(line))  for line in data_file_test_text.read().splitlines()]
    data_file_test_labels =open(testing_path_labels, "r", encoding='utf-8')
    datatable_arabic_test_labels = [int(line) for line in data_file_test_labels.read().splitlines()]
    d3 = {'tweet': datatable_arabic_test_text, 'label': datatable_arabic_test_labels}
    df_test = pd.DataFrame(data=d3)
    
    return df_train, df_valid, df_test