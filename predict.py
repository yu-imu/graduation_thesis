from gensim.models import word2vec

model = word2vec.Word2Vec.load("./wiki.model")
results = model.wv.most_similar(positive=[''])
for result in results:
    print(result)


[['users','TABLE','set','where','/*','update','delete','if','null',
'account_table','`','1','CONCAT','performance_schema.events_waits_summary_by_instance',
'ALL','WHEN','0','EXISTS','NULL','EXISTS',',','WHEN','ELSE','SELECT','END','CASE','CONCAT','THEN',',0x716a717a71']]
