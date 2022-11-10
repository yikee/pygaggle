from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
import json

with open('whatsthatbook_results') as f:
    data = json.loads(f.read())
print("there are %s query examples" % len(data))
print("top %s rank" % len(data[0]['ctxs']))

res = []
reranker =  MonoT5()
for instance in data:
    query = Query(instance['question'])
    ctxs = instance['ctxs']
    passages = [[item['id'][5:], item['text']] for item in ctxs]
    texts = [Text(p[1], {'docid': p[0]}, 0) for p in passages]
    res.append(reranker.rerank(query, texts))

with open("MonoT5_ranking", "w") as outfile:
    json.dump(res, outfile)