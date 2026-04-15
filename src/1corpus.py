"""
STEP 1 (Local Experiment)
Preparing corpus embeddings. Using biencoder embedding model 
BAAI/bge-base-en-v1.5
"""

"""
Generating embeddings 
"""

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "BAAI/bge-base-en-v1.5"
OUTPUT_FILE = "data/corpus.npy"

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading MS MARCO corpus")

dataset = load_dataset("ms_marco", "v1.1", split="train")

#passages = dataset["passage"]
passages = []
for row in dataset.select(range(50000)):
        passages.extend(row["passages"]["passage_text"])
        relevant = [i for i, flag in enumerate(row["passages"]["is_selected"]) if flag]

print("Total passages:", len(passages))


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def embed(texts):

    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**tokens)

    embeddings = outputs.last_hidden_state[:, 0]

    embeddings = torch.nn.functional.normalize(
        embeddings,
        p=2,
        dim=1
    )

    return embeddings.cpu().numpy()


vectors = []

for i in tqdm(range(0, len(passages), BATCH_SIZE)):

    batch = passages[i:i+BATCH_SIZE]

    emb = embed(batch)

    vectors.append(emb)


vectors = np.vstack(vectors)

np.save(OUTPUT_FILE, vectors)

print("Saved corpus embeddings:", vectors.shape)