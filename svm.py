# 次にやるべきこと
# 1 encodingの事故をなくす 最重要
# 2 時間を計測できるようにする

import MeCab
import csv
import codecs
import time
import numpy as np
import pandas as pd
from gensim import corpora, matutils
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mecab = MeCab.Tagger('mecabrc')


def tokenize(text):
    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next


def get_words(contents):
    ret = []
    # ただのリストを投げる場合はこのタイプにする
    for content in contents:
        ret.append(get_words_main(content))
    return ret


def get_words_main(content):
    return [token for token in tokenize(content)]

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
    start = time.time()
    column = []
    num = []
    with open('word.csv', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader)  # ヘッダーを読み飛ばしたい時

        for row in reader:
            column.append(row[0])
            num.append(row[1])

    data_train_s, data_test_s, label_train_s, label_test_s = train_test_split(column, num, test_size=0.3)
    # print(data_train_s)
    words = get_words(data_train_s)

    dictionary = corpora.Dictionary(words)
    dictionary.filter_extremes(no_below=3, no_above=0.6)
    # docを取り出してループする
    result = get_bow_words(dictionary, words)


    # 学習させる
    classifier = SVC()
    classifier.fit(result, label_train_s)
    test_words = get_bow_words(dictionary,get_words(data_test_s))
    result = classifier.predict(test_words)

    accuracy_test = accuracy_score(result, label_test_s)

    elapsed_time = time.time() - start
    #import pdb; pdb.set_trace()
    f = open('test.csv','a')

    f.write('テストデータに対する正解率： %.2f' % accuracy_test)
    f.write('\n')
    f.write("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    f.write('\n')
    f.close()
