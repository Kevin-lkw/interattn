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


# ---------- helpers ----------
def _softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def _effective_rank_from_singular_values(s, eps=1e-12):
    # effective rank = exp(entropy of normalized singular values)
    p = s / (np.sum(s) + eps)
    H = -np.sum(p * np.log(p + eps))
    return float(np.exp(H))

def _cluster_low_rank_stats(K_np, labels, energy_thr=0.90):
    """
    K_np: [seq, d]  (建议用未normalize的K，用来衡量能量/秩更合理)
    labels: [seq]
    returns: sizes, erank, r_energy (r@energy_thr)
    """
    C = int(labels.max() + 1)
    sizes = np.bincount(labels, minlength=C).astype(int)
    erank = np.zeros(C, dtype=np.float64)
    r_energy = np.zeros(C, dtype=np.int32)

    for c in range(C):
        idx = np.where(labels == c)[0]
        if idx.size <= 1:
            erank[c] = 1.0
            r_energy[c] = 1
            continue

        Xc = K_np[idx]
        Xc = Xc - Xc.mean(axis=0, keepdims=True)

        # SVD: Xc = U S V^T
        # singular values s: [min(|C|, d)]
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)

        erank[c] = _effective_rank_from_singular_values(s)
        cum = np.cumsum(s**2) / (np.sum(s**2) + 1e-12)
        r_energy[c] = int(np.searchsorted(cum, energy_thr) + 1)

    return sizes, erank, r_energy

def _cluster_attention_concentration(Q_np, K_np, labels, query_idx, topm_list=(1,2,4)):
    """
    Q_np: [seq, d] (真实q)
    K_np: [seq, d] (真实k)  (这里使用after_rope的k更一致)
    labels: [seq] (cluster on keys)
    query_idx: 1D array of query positions (subsample for speed)
    Returns: dict with entropy array and topm mass arrays
    """
    C = int(labels.max() + 1)
    members = [np.where(labels == c)[0] for c in range(C)]

    ent = []
    intra_ent = []  
    topm_mass = {m: [] for m in topm_list}
    intra_topk_mass = {m: [] for m in topm_list}
    for t in query_idx:
        # logits over all keys: [seq]
        logits = K_np @ Q_np[t]  # dot product
        a = _softmax_np(logits[None, :], axis=1).reshape(-1)

        # aggregate to clusters
        ac = np.zeros(C, dtype=np.float64)
        for c in range(C):
            idx = members[c]
            if idx.size > 0:
                ac[c] = a[idx].sum()
        # print("max group",ac.argmax(),ac.max())
        # entropy
        p = ac / (np.sum(ac) + 1e-12)
        ent.append(float(-np.sum(p * np.log(p + 1e-12))))
        # import ipdb; ipdb.set_trace()
        # top-m mass
        srt = np.sort(ac)[::-1]
        for m in topm_list:
            topm_mass[m].append(float(np.sum(srt[:m])))
        
        ## intra cluster mass ratio (optional)
        c_star = int(np.argmax(ac))
        idx = members[c_star]
        mass = float(ac[c_star]) + 1e-12

        # conditional within cluster
        within = a[idx] / mass  # sums to 1 over idx
        intra_ent.append(float(-np.sum(within * np.log(within + 1e-12))))
        srt = np.sort(within)[::-1]
        for m in topm_list:
            intra_topk_mass[m].append(float(np.sum(srt[:m])))


    return {
        "entropy": np.array(ent, dtype=np.float64),
        "topm_mass": {m: np.array(v, dtype=np.float64) for m, v in topm_mass.items()},
        "intra_entropy": np.array(intra_ent, dtype=np.float64), 
        "intra_topk_mass": {m: np.array(v, dtype=np.float64) for m, v in intra_topk_mass.items()},
        "num_clusters": C,
        "num_queries": len(query_idx),
    }

def clustering(layer_idx, head_idx=0, after_rope=True, key_type='k', n_clusters=5, normalize=True, use_tsne=False):
    if after_rope:
        key = rope_qkv[layer_idx][key_type] # [B,nh,seq_len,hd]
        specific_head_key = key[0, head_idx, :, :] # [seq_len, head_dim]
    else:
        key = kv_info[f"model.layers.{layer_idx}.self_attn.{key_type}_proj"]
        seq_len = key.shape[1]
        hidden_size = model_config.hidden_size
        num_heads = model_config.num_attention_heads
        head_dim = hidden_size // num_heads
        key_heads = key.view(seq_len, num_heads, head_dim)
        specific_head_key = key_heads[:, head_idx, :]  # [seq_len, head_dim]
    key_np = specific_head_key.float().cpu().numpy()
    if normalize:
        key = F.normalize(specific_head_key.float(), dim=-1).cpu().numpy()
    else:
        key = specific_head_key.float().cpu().numpy()
    labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(key)
    print("Cluster sizes:")
    print(np.bincount(labels))

    # ===== Dim reduction for visualization =====
    if use_tsne:
        emb = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto"
        ).fit_transform(key)
    else:
        emb = PCA(n_components=2).fit_transform(key)
    out_dir=f"img/layer{layer_idx}/head{head_idx}/clusters{n_clusters}/"
    os.makedirs(out_dir, exist_ok=True)
    # ===== Plot 1: embedding colored by cluster =====
    plt.figure(figsize=(6,5))
    plt.scatter(emb[:,0], emb[:,1], c=labels, s=5, cmap="tab20")
    plt.title("Cluster embedding")
    plt.colorbar()
   
    plt.savefig(f"{out_dir}embedding.png")

    # ===== Plot 2: cluster vs token position =====
    plt.figure(figsize=(10,3))
    plt.scatter(np.arange(len(labels)), labels, s=3)
    plt.title("Cluster ID vs Token Position")
    plt.xlabel("Token index")
    plt.ylabel("Cluster")
    plt.savefig(f"{out_dir}cluster_vs_tokenpos.png")
    
    # ===== Low-rank stats per cluster =====
    sizes, erank, r90 = _cluster_low_rank_stats(key_np, labels, energy_thr=0.90)

    # scatter: size vs effective rank
    plt.figure(figsize=(6, 4))
    plt.scatter(sizes, erank, s=20)
    plt.xlabel("Cluster size")
    plt.ylabel("Effective rank")
    plt.title(f"Effective rank vs size (L{layer_idx} H{head_idx})")
    plt.tight_layout()
    plt.savefig(f"{out_dir}lowrank_effective_rank_vs_size.png", dpi=200)

    # scatter: size vs r@90
    plt.figure(figsize=(6, 4))
    plt.scatter(sizes, r90, s=20)
    plt.xlabel("Cluster size")
    plt.ylabel("r @ 90% energy")
    plt.title(f"r@90% energy vs size (L{layer_idx} H{head_idx})")
    plt.tight_layout()
    plt.savefig(f"{out_dir}lowrank_r90_vs_size.png", dpi=200)

    if not after_rope:
        raise ValueError("cluster-attention concentration expects after_rope=True (needs real Q,K after RoPE).")

    # get real Q and K (after_rope) for attention computation
    Q = rope_qkv[layer_idx]["q"][0, head_idx].float().cpu().numpy()  # [seq, d]
    # Q_dummy = np.random.randn(*Q.shape).astype(np.float32)
    K = rope_qkv[layer_idx]["k"][0, head_idx].float().cpu().numpy()  # [seq, d]

    seq = K.shape[0]
    q_subsample = 1024
    if q_subsample is None or q_subsample >= seq:
        query_idx = np.arange(seq)
        
    else:
        rng = np.random.default_rng(0)
        # query_idx = np.sort(rng.choice(seq, size=q_subsample, replace=False))
        # use last 256 tokens as queries for reproducibility
        query_idx = np.arange(seq - q_subsample, seq)

    stats = _cluster_attention_concentration(
        Q_np=Q,
        K_np=K,
        labels=labels,
        query_idx=query_idx,
        topm_list=[1,2,4,8],
    )

    # entropy hist
    plt.figure(figsize=(6, 4))
    plt.hist(stats["entropy"], bins=50)
    plt.title(f"q-attn entropy (L{layer_idx} H{head_idx}, k={n_clusters})")
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_dir}q_attn_entropy_hist.png", dpi=200)
    # intra-cluster entropy hist
    plt.figure(figsize=(6, 4))
    plt.hist(stats["intra_entropy"], bins=50)
    plt.title(f"q-attn intra-cluster entropy (L{layer_idx} H{head_idx}, k={n_clusters})")
    plt.xlabel("Intra-cluster Entropy")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_dir}q_attn_intra_entropy_hist.png", dpi=200)
    
    # top-m mass hists
    for m, arr in stats["topm_mass"].items():
        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=50)
        plt.title(f"Top-{m} cluster mass (L{layer_idx} H{head_idx}, k={n_clusters})")
        plt.xlabel("Mass")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{out_dir}q_attn_top{m}_mass_hist.png", dpi=200)
    # intra cluster top-k mass hists
    for m, arr in stats["intra_topk_mass"].items():
        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=50)
        plt.title(f"Top-{m} intra-cluster mass (L{layer_idx} H{head_idx}, k={n_clusters})")
        plt.xlabel("Mass")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{out_dir}q_attn_intra_top{m}_mass_hist.png", dpi=200)
    plt.close('all')
    return stats
    
# for num_clusters in [16,32,64]:

#     highest_ent_layer = -1
#     highest_ent_head = -1
#     for layer_idx in range(0,32):
#         highest_ent = -1
#         for head_idx in range(0, model_config.num_attention_heads):
#             stats = clustering(
#                 layer_idx=layer_idx, 
#                 head_idx=head_idx, 
#                 after_rope=True, 
#                 key_type='k', 
#                 n_clusters=num_clusters, 
#                 use_tsne=True, 
#             )
#             if stats["entropy"].max() > highest_ent:
#                 highest_ent = stats["entropy"].max()
#                 highest_ent_head = head_idx
#         print(f"Clustering with {num_clusters} clusters: \n \
            # Highest entropy observed: {highest_ent} (Layer {layer_idx}, Head {highest_ent_head})")

num_clusters = 64
stats = clustering(
                layer_idx=8, 
                head_idx=20, 
                after_rope=True, 
                key_type='k', 
                n_clusters=num_clusters, 
                use_tsne=True, 
            )