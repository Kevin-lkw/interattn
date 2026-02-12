import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import os
import math
model = "llama-7b-hf"
dataset_name="wikitext"
start = 0
kv = torch.load(f"{model}_{dataset_name}_st{start}.pt", weights_only=False)

model_config = kv["model_config"]
kv_info = kv["before_rope"]
rope_qkv = kv["after_rope"]

def get_attention_map_after_rope(layer_idx, head_idx=0, causal=True, dtype=torch.float32):
    """
    返回: attn [seq_len, seq_len] (softmax 后)
    """
    Q = rope_qkv[layer_idx]['q']  # [B, nh, seq, hd]
    K = rope_qkv[layer_idx]['k']  # [B, nh, seq, hd]

    q = Q[0, head_idx].to(dtype)  # [seq, hd]
    k = K[0, head_idx].to(dtype)  # [seq, hd]

    hd = q.shape[-1]
    scores = (q @ k.T) / math.sqrt(hd)  # [seq, seq]

    if causal:
        seq = scores.shape[0]
        mask = torch.triu(torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    return attn

def analyze_similarity(layer_idx, head_idx=0, key_type='k'):
    # 假设我们取第 layer_idx 层的 K 向量
    # key 的形状可能是 [1, seq_len, hidden_size]
    key = kv_info[f"model.layers.{layer_idx}.self_attn.{key_type}_proj"]
    
    seq_len = key.shape[1]
    hidden_size = model_config.hidden_size
    num_heads = model_config.num_attention_heads
    head_dim = hidden_size // num_heads
    
    key_heads = key.view(seq_len, num_heads, head_dim)
    x = key_heads[:, head_idx, :].float()
    x = F.normalize(x, dim=-1)                                   
    sim_matrix2 = x @ x.T
    
    return sim_matrix2

def analyze_similarity_after_rope(layer_idx, head_idx=0, query_type='k', key_type='k'):
    query = rope_qkv[layer_idx][query_type] # [B,nh,seq_len,hd]
    key = rope_qkv[layer_idx][key_type] # [B,nh,seq_len,hd]
    specific_head_query = query[0, head_idx, :, :] # [seq_len, head_dim]
    specific_head_key = key[0, head_idx, :, :] # [seq_len, head_dim]
    Nq = F.normalize(specific_head_query.float(), dim=-1)  # [seq_len, head_dim]
    Nk = F.normalize(specific_head_key.float(), dim=-1)    # [seq_len, head_dim]
    sim_matrix = Nq @ Nk.T  # [seq_len, seq_len]
    return sim_matrix

def plot_similarity_lower_triangle(layer_idx, head_idx, after_rope=True, query_type='k', key_type='k'):
    sim_matrix = analyze_similarity_after_rope(layer_idx, head_idx, query_type=query_type, key_type=key_type) 
    # 将 tensor 转为 numpy
    data = sim_matrix.detach().cpu().numpy()
    
    # 创建掩码：只保留下三角 (including diagonal)
    mask = np.triu(np.ones_like(data, dtype=bool), k=1)
    
    plt.figure(figsize=(10, 8))
    
    # 使用 RdBu_r: 红色代表 1.0 (高相似度), 蓝色代表低相似度
    # center=0.5 可以让颜色过渡更符合直觉
    ax = sns.heatmap(
        data, 
        mask=mask, 
        cmap="RdBu_r", 
        vmin=-1, vmax=1,  # 相似度在 -1 到 1 之间
        center=0, 
        linewidths=0, 
        cbar_kws={"label": "Cosine Similarity"}
    )
    
    plt.title(f"KV Similarity (Lower Triangle)\nLayer {layer_idx}, Head {head_idx}")
    plt.xlabel("Token Index")
    plt.ylabel("Token Index")
    out_dir = f"./img/layer{layer_idx}/head{head_idx}/"
    os.makedirs(out_dir, exist_ok=True)
    if after_rope:
        out_dir += "after_rope.png"
    else:
        out_dir += "before_rope.png"
        
    plt.savefig(out_dir)
    
def plot_similarity_grid(layers, num_heads_to_plot=8, query_type='q', key_type='k'):
    num_layers = len(layers)
    # 设置画布大小：每张子图 4x4 英寸
    fig, axes = plt.subplots(num_layers, num_heads_to_plot, 
                             figsize=(num_heads_to_plot * 4, num_layers * 4),
                             constrained_layout=True)
    
    for i, layer_idx in enumerate(layers):
        for head_idx in range(num_heads_to_plot):
            print(f"Plotting Layer {layer_idx}, Head {head_idx}...")
            ax = axes[i, head_idx]
            
            # 计算相似度 (After RoPE)
            # before rope
            sim_matrix = analyze_similarity_after_rope(layer_idx, head_idx, query_type=query_type, key_type=key_type)
            data = sim_matrix.detach().cpu().numpy()
            
            # 创建掩码
            mask = np.triu(np.ones_like(data, dtype=bool), k=1)
            
            # 绘图：去掉 linewidths 以提升 4096 尺度的渲染速度
            sns.heatmap(
                data, 
                mask=mask, 
                cmap="RdBu_r", 
                vmin=-1, vmax=1, 
                center=0, 
                cbar=(head_idx == num_heads_to_plot - 1), # 只在每行最后一列画 colorbar
                ax=ax,
                xticklabels=False, # 4096 个标签太挤了，关掉
                yticklabels=False,
                square=True
            )
            
            if i == 0:
                ax.set_title(f"Head {head_idx}", fontsize=15)
            if head_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=15, fontweight='bold')


    plt.suptitle(f"KV Similarity (After RoPE) - Context Length: {data.shape[0]}", fontsize=20)
    out_path = f"./img/kv_sim_grid_comparison_{key_type}.png"
    os.makedirs("./img/", exist_ok=True)
    plt.savefig(out_path, dpi=150) # 4096 尺度下 150 dpi 足够看清结构
    print(f"Grid plot saved to {out_path}")

def plot_length_distribution(layer_index, head_index, key_type='k'):
    K = rope_qkv[layer_index][key_type] # [B,nh,seq_len,hd]
    k_vectors = K[0, head_index, :, :].detach().cpu().numpy() # [seq_len, head_dim]
    # 计算每个 token 的 key 向量的 L2 范数
    lengths = np.linalg.norm(k_vectors, axis=1) # [seq_len]
    plt.figure(figsize=(10, 6))
    plt.plot(lengths)
    plt.title(f"Key Vector Lengths - Layer {layer_index}, Head {head_index}")
    plt.xlabel("Token Position")
    plt.ylabel("L2 Norm of Key Vector")
    out_dir = f"./img/layer{layer_index}/head{head_index}/"
    os.makedirs(out_dir, exist_ok=True)
    file_name = out_dir + f"{key_type}_length_distribution.png"
    plt.savefig(file_name)

def plot_attention_grid(layers, num_heads_to_plot, causal=True, save_path="./img/attn_grid.png", dpi=150, zoom=None):

    num_layers = len(layers)

    fig, axes = plt.subplots(
        num_layers, num_heads_to_plot,
        figsize=(num_heads_to_plot * 4, num_layers * 4),
        constrained_layout=True
    )

    # 兼容 num_layers==1 或 num_heads_to_plot==1 时 axes 不是 2D 的情况
    if num_layers == 1 and num_heads_to_plot == 1:
        axes = np.array([[axes]])
    elif num_layers == 1:
        axes = np.array([axes])
    elif num_heads_to_plot == 1:
        axes = np.array([[ax] for ax in axes])

    last_im = None

    for i, layer_idx in enumerate(layers):
        for j in range(num_heads_to_plot):
            head_idx = j
            print(f"Plotting Attention Map - Layer {layer_idx}, Head {head_idx}...")

            attn = get_attention_map_after_rope(layer_idx, head_idx, causal=causal)
            a = attn.detach().cpu().numpy()  # [seq, seq]

            if zoom is not None:
                s, e = zoom
                a = a[s:e, s:e]

            ax = axes[i, j]
            last_im = ax.imshow(
                a,
                aspect="auto",
                cmap="Reds",
                vmin=0.0, vmax=1.0,
                interpolation="nearest"
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(f"Head {head_idx}", fontsize=14)
            if j == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=14, fontweight="bold")

    # 全局一个 colorbar（推荐）
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label("Attention Weight")

    title = f"Attention Map Grid (After RoPE) - causal={causal}"
    if zoom is not None:
        title += f" - zoom={zoom[0]}:{zoom[1]}"
    fig.suptitle(title, fontsize=18)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print("Saved grid:", save_path)
        
# plot_similarity_lower_triangle(layer_idx=24, head_idx=5, after_rope=True, key_type='v')
# plot_similarity_grid(layers=[8,12,24], num_heads_to_plot=4, key_type = 'k')    
# plot_length_distribution(layer_index=24, head_index=5, key_type='q')
plot_attention_grid(layers=[4,8,12,24,30], num_heads_to_plot=6, causal=True, save_path="./img/attn_grid.png", dpi=150,zoom=(0,128))