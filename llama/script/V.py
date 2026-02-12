import torch
import torch.nn.functional as F
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from matplotlib import pyplot as plt
import os
model = "llama-7b-hf"
dataset_name="wikitext"
start = 0
kv = torch.load(f"{model}_{dataset_name}_st{start}.pt", weights_only=False)

model_config = kv["model_config"]
kv_info = kv["before_rope"]
rope_qkv = kv["after_rope"]
Wo = kv["Wo"]
Wlm = kv["Wlm"]

V = rope_qkv[-1]["v"]  # ()