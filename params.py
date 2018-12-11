# 次にやるべきこと
# 1 encodingの事故をなくす 最重要
# # -*- coding:utf-8 -*-2 時間を計測できるようにする
# -*- coding:utf-8 -*-
import csv
import codecs
import nltk
from gensim import corpora, matutils, models
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

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
    column = []
    num = []
    with open('word.csv', errors='ignore') as f:
         reader = csv.reader(f)
         header = next(reader)  # ヘッダーを読み飛ばしたい時

         for row in reader:
             column.append(row[0])
             num.append(row[1])

    data_train_s, data_test_s, y_train, y_test= train_test_split(column, num, test_size=0.4, random_state=0)

    train_words = get_words(data_train_s)
    test_words = get_words(data_test_s)
    dictionary = corpora.Dictionary(train_words)
    # docを取り出してループする
    X_train = get_bow_words(dictionary, train_words)
    X_test = get_bow_words(dictionary, test_words)
    ## チューニングパラメータ
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['accuracy', 'precision', 'recall']

    for score in scores:
        print('\n' + '='*50)
        print(score)
        print('='*50)

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score, n_jobs=-1)
        clf.fit(X_train, y_train)

        print("\n+ ベストパラメータ:\n")
        print(clf.best_estimator_)
        print(clf.best_params_)
        print(clf.best_score_)
