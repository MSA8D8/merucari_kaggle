#  code:utf-8
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords


def get_keras_data(dataset,MAX_NAME_SEQ,MAX_ITEM_DESC_SEQ):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        ,'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ)
        ,'brand_name': np.array(dataset.brand_name)
        ,'category_name': np.array(dataset.category_name)
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset[["shipping"]])
    }
    return X

def label_encode(train,test):
    le = LabelEncoder()

    le.fit(np.hstack([train.category_name, test.category_name]))
    train.category_name = le.transform(train.category_name)
    test.category_name = le.transform(test.category_name)

    le.fit(np.hstack([train.brand_name, test.brand_name]))
    train.brand_name = le.transform(train.brand_name)
    test.brand_name = le.transform(test.brand_name)
    del le
    return train, test


def description_vectorizer(text):
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=1,
                                 max_features=10,
                                 stop_words=stopwords.words('english'))
    tfidf_weighted_matrix = vectorizer.fit_transform(text)
    return pd.DataFrame(tfidf_weighted_matrix.toarray()), vectorizer.get_feature_names()


def text_process(train, test):
    raw_text = np.hstack([train.item_description.str.lower(),
                          train.name.str.lower()])
    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)
    train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
    test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
    train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
    test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
    return train, test

def item_description_tokenizer(text):
    word_list = [sentence.split(' ') for sentence in text]
    word_list = [[re.sub(re.compile("[!-/:-@[-`{-~]"), '', word)
                 for word in sentence] for sentence in word_list]

    word_list = [[re.sub(re.compile("[0-9]+"),'',word) for word in sentence] for sentence in word_list]
    word_list = [[word for word in filter(lambda s:s!='', sentence)] for sentence in word_list]
    sentences = [' '.join(sentence) for sentence in word_list]
    return sentences
