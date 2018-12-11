# -graduation_thesis

split_data_train_s = get_words(data_train_s)
model = word2vec.Word2Vec(split_data_train_s, size=100,min_count=5,window=5,iter=100)
model.save("./wiki.model")
