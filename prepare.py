# 次にやるべきこと
# 1 encodingの事故をなくす 最重要
# # -*- coding:utf-8 -*-2 時間を計測できるようにする
# -*- coding:utf-8 -*-
import csv
import nltk
import numpy as np
from gensim import corpora
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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

if __name__ == '__main__':
    # デコードでたまに失敗する
    column = []
    num = []
    with open('word.csv', errors='ignore') as f:
         reader = csv.reader(f)
         header = next(reader)  # ヘッダーを読み飛ばしたい時

         for row in reader:
             column.append(row[0])
             num.append(row[1])
    #６割学習する
    data_train_s, data_test_s, label_train_s, label_test_s = train_test_split(column, num, test_size=0.4)
    # 辞書作成 TF-IDFは微妙かも
    texts = [[ '(',')',';',' ']]
    words = get_words(data_train_s)
    dictionary = corpora.Dictionary(words)
    dictionary.filter_extremes(no_above=0.8)
    dictionary.add_documents(texts)
    dictionary.save_as_text('aaa.txt')
