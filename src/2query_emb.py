"""
STEP 2:  (Local Experiment)

Building the experiment query embedding
"""

import json
import os  
import random 

import numpy as np  
from datasets import load_dataset 
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModel 
import torch 
#print(torch.cuda.is_available())


#define paths 
MODEL_NAME = "BAAI/bge-base-en-v1.5"
OUTPUT_FILE = "data/metadata_queries.npy"
META_FILE = "data/metadata_queries.json"

#set
BATCH_SIZE = 64
NUM_QUERIES = 5000
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("data", exist_ok=True) #directory check 

#check data
print("Loading MS MARCO train split..")
dataset = load_dataset("ms_marco", "v1.1", split="train")

print("collecting queries....")
queries = []
for row in dataset.select(range(NUM_QUERIES)):
    q = row['query']
    if q and isinstance(q, str) and q.strip():
        queries.append(q.strip())

print(f"collected {len(queries)} queries")


#normalize and train
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

def embed(texts):
    tokens = tokenizer(
        texts,
        padding = True,
        truncation=True,
        max_length = 256, 
        return_tensors = "pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**tokens)

    emb = outputs.last_hidden_state[:, 0]
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

#save queries 
all_vec = []
for i in tqdm(range(0, len(queries), BATCH_SIZE), desc="Ebedding queries"):
    batch = queries[i:i + BATCH_SIZE]
    all_vec.append(embed(batch))

queries_np = np.vstack(all_vec).astype("float32")
np.save(OUTPUT_FILE, queries_np)

with open(META_FILE, "w") as f:
    json.dump({
        "model_name": MODEL_NAME,
        "num_queries": int(len(queries)),
        "embedding_dim": int(queries_np.shape[1]),
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
    }, 
    f, 
    indent=2,
    )
print(f'Saved query embeddings to {OUTPUT_FILE} with shape {queries_np.shape}')