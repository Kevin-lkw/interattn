import token
from turtle import left
from sympy import per
from tokenizers import InputSequence
import torch
import torch.nn.functional as F
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from matplotlib import pyplot as plt
import os
from transformers import AutoTokenizer

llama_model = "meta-llama/Llama-2-7b-hf"
model = "llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(
    llama_model,
    use_fast=False,        
)
dataset_name="wikitext"
start = 0
kv = torch.load(f"../{model}_{dataset_name}_st{start}.pt", weights_only=False)



model_config = kv["model_config"]
kv_info = kv["before_rope"]
rope_qkv = kv["after_rope"]
Wo = kv["Wo"] # Shape (hidden_size, hidden_size)
Wlm = kv["Wlm"] # Shape (vocab_size, hidden_size)
inputs = kv["input"]

L = model_config.num_hidden_layers

from analyse_k import get_attention_map_after_rope
attention_score = {}
per_token_attention = torch.zeros(inputs['input_ids'].shape[1])  # [seq_len]
for head_idx in range(model_config.num_attention_heads):
    attention_score[head_idx] = get_attention_map_after_rope(layer_idx=L-1, head_idx=head_idx, causal=True)
    per_token_attention = per_token_attention + attention_score[head_idx].sum(dim=0)  # 每个 token 的总 attention 权重


V:torch.Tensor = rope_qkv[L-1]["v"]  # last layer's value, shape (B,nh,seq_len,hd)
V = V.transpose(1, 2).contiguous()  # (B,seq_len,nh,hd)
B, seq_len, nh, hd = V.shape
V = V.view(B, seq_len, nh * hd)  # (B,seq_len,nh*hd)

Z_hidden = V @ Wo.T  # (B,seq_len,hidden_size)
Z_final = Z_hidden @ Wlm.T  # (B,seq_len, vocab_size)
Z_hidden_np = Z_hidden[0].cpu().numpy()  # (seq_len, hidden_size)
Z_hidden_np = Z_hidden_np.astype(np.float32)

# print("Output shape:", Z_final.shape)

# # SVD for Z_hidden

# Z_centered = Z_hidden_np - Z_hidden_np.mean(axis=0)  # Centering
# # import ipdb; ipdb.set_trace() 
# U, S, VT = np.linalg.svd(Z_centered, full_matrices=False)

# # Plot singular values
# plt.figure(figsize=(8, 6))
# plt.plot(S, marker='o')
# plt.yscale('log')
# plt.title("Singular Values of Z_hidden")
# plt.xlabel("Index")
# plt.ylabel("Singular Value(log scale)")
# plt.grid()
# path = "../img/v_analysis"
# os.makedirs(path, exist_ok=True)
# plt.savefig(f"{path}/Z_hidden_singular_values_{model}_{dataset_name}_st{start}.png")

# # Effective rank
# energy = np.cumsum(S**2) / np.sum(S**2)
# effective_rank = np.searchsorted(energy, 0.9) + 1
# print(f"Effective rank of Z_hidden: {effective_rank}")


# # PCA for Z_hidden
# pca = PCA(n_components=2)
# Z_pca = pca.fit_transform(Z_hidden_np)  # (seq_len, 2)

# plt.figure(figsize=(8, 6))
# plt.scatter(Z_pca[:, 0], Z_pca[:, 1], s=5)
# plt.title("PCA of Z_hidden")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.grid()
# path = "../img/v_analysis"
# os.makedirs(path, exist_ok=True)
# plt.savefig(f"{path}/Z_hidden_PCA_{model}_{dataset_name}_st{start}.png")



"""
We want to analyze the convex hull of the value vectors Z.
where Z = V @ Wo.T.
"""

# # support vectors 
# num_dirctions = 2048
# n,d = Z_hidden_np.shape
# cnt = np.zeros(n)
# for i in range(num_dirctions):
#     r = np.random.randn(d)
#     r = r / np.linalg.norm(r)
#     projections = Z_hidden_np @ r  # (n,)
#     argmax = np.argmax(projections)
#     cnt[argmax] += 1

# # print the number of positive counts
# num_support_vectors = np.sum(cnt > 0)
# print(f"Number of support vectors in Z_hidden: {num_support_vectors} out of {n}")
# print("top 10 support vectors indices and counts:")
# top_10_indices = np.argsort(cnt)[-10:][::-1]
# for idx in top_10_indices:
#     print(f"Index: {idx}, Count: {cnt[idx]}")

# # plot the relation between counts and per token attention
# plt.figure(figsize=(8, 6))
# plt.scatter(cnt, per_token_attention.cpu().numpy(), alpha=0.6)
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Support Vector Counts ")
# plt.ylabel("Per Token Attention")
# plt.title("Support Vector Counts vs Per Token Attention")
# plt.grid()
# plt.savefig(f"../img/v_analysis/support_vector_counts_vs_attention_{model}_{dataset_name}_st{start}.png")


"""
finding the optimal alpha star for a given token
"""
def solve_alpha_star_logits(Z_logit: torch.Tensor, y_id: int, steps=500, lr=0.05):
    """
    Z_logit: (n, V) float32
    y_id: int
    returns alpha_star (n,)
    """
    device = Z_logit.device
    n, V = Z_logit.shape
    w = torch.zeros(n, device=device, requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr)

    for _ in range(steps):
        alpha = F.softmax(w, dim=0)              # (n,)
        s = alpha @ Z_logit                      # (V,)
        loss = -s[y_id] + torch.logsumexp(s, dim=0)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return F.softmax(w, dim=0).detach()

