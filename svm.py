# 次にやるべきこと
# 1 encodingの事故をなくす 最重要
# # -*- coding:utf-8 -*-2 時間を計測できるようにする
# -*- coding:utf-8 -*-
import MeCab
import csv
import codecs
import time
import numpy as np
import pandas as pd
import nltk
from gensim import corpora, matutils, models
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mecab = MeCab.Tagger('mecabrc')


def tokenize_jp(text):
    #ニホン語版形態素解析
    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next

def tokenize_en(text):
    #英語版
    li_uniq = list(set(nltk.word_tokenize(text)))
    return li_uniq


def get_words(contents):
    ret = []
    # ただのリストを投げる場合はこのタイプにする
    for content in contents:
        #変更
        ret.append(tokenize_en(content))
    return ret


def get_words_main(content):
    # 日本語版
    return [token for token in tokenize_jp(content)]

def get_bow_words(dictionary, words):
    # 特徴ベクトル化
    doc_model = []
    for word in words:
        tmp = dictionary.doc2bow(word)
        dense = list(matutils.corpus2dense([tmp], num_terms=len(dictionary)).T[0])
        doc_model.append(dense)
    return doc_model

if __name__ == '__main__':
    # デコードでたまに失敗する
    i=1
    while i<101:
        start = time.time()
        column = []
        num = []
        with open('word_double.csv', errors='ignore') as f:
             reader = csv.reader(f)
             header = next(reader)  # ヘッダーを読み飛ばしたい時

             for row in reader:
                 column.append(row[0])
                 num.append(row[1])

        data_train_s, data_test_s, label_train_s, label_test_s = train_test_split(column, num, test_size=0.4)

        words = get_words(data_train_s)
        dictionary = corpora.Dictionary(words)
        dictionary = corpora.Dictionary.load_from_text('aaa.txt')
        train_result = get_bow_words(dictionary, words)


        # 学習させる C': 1, 'gamma': 0.001, 'kernel': 'rbf'
        classifier = SVC(C= 1, kernel='rbf', gamma= 0.001)
        classifier.fit(train_result, label_train_s)
        test_words = get_bow_words(dictionary,get_words(data_test_s))
        result = classifier.predict(test_words)
        accuracy_test = accuracy_score(result, label_test_s)

        elapsed_time = time.time() - start
        #import pdb; pdb.set_trace()
        f = open('test.csv','a')

        f.write('%.5f' % accuracy_test)
        f.write('\n')
        f.write("{0}".format(elapsed_time))
        f.write('\n')
        f.close()
        i = i+1
