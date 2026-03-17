import numpy as np
import faiss
import time
import pandas as pd

CORPUS_PATH="corpus.npy"
QUERY_PATH="queries.npy"
GT_PATH="gt_top100.npy"

K=100
QUERIES=200
WARMUP=20

M=32
EF_CONSTRUCTION=200
EF_SEARCH=100


def recall_at_k(I,gt,k):
    hits=0
    for i in range(len(I)):
        hits+=len(set(I[i][:k]).intersection(set(gt[i][:k])))
    return hits/(len(I)*k)


xb=np.load(CORPUS_PATH).astype('float32')
xq=np.load(QUERY_PATH).astype('float32')
gt=np.load(GT_PATH)

d=xb.shape[1]

index=faiss.IndexHNSWFlat(d,M)
index.hnsw.efConstruction=EF_CONSTRUCTION
index.hnsw.efSearch=EF_SEARCH

print("building HNSW")
index.add(xb)

records=[]

for i in range(WARMUP):
    index.search(xq[i:i+1],K)

for i in range(WARMUP,QUERIES+WARMUP):

    faiss.cvar.hnsw_stats.reset()

    start=time.time()
    D,I=index.search(xq[i:i+1],K)
    latency=(time.time()-start)*1000

    nodes=faiss.cvar.hnsw_stats.nhops
    candidates=faiss.cvar.hnsw_stats.ndis

    r10=recall_at_k(I,gt[i:i+1],10)
    r100=recall_at_k(I,gt[i:i+1],100)

    records.append({
        "query":i,
        "latency_ms":latency,
        "nodes_visited":nodes,
        "candidates":candidates,
        "recall@10":r10,
        "recall@100":r100
    })

df=pd.DataFrame(records)
df.to_csv("results_hnsw.csv",index=False)

print(df.describe())