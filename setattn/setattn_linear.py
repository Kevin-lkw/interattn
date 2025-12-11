# -*- coding: utf-8 -*-
"""
Custom Linear Attention and Set Attention modules using FLA library.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple

from fla.layers.linear_attn import LinearAttention
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn
import os

def get_sets(T: int, levelrand: bool, level: int, levelmax: int, set_policy: str, device):
    sets = []
    setlevel = torch.randint(0,levelmax+1,()) if levelrand else level
    if set_policy == "small":
        """
        for smaller sets, we can directly set level to levelmax when setlevel > levelmax
        but this is not for larger sets, as when level > levelmax, there will be no sets and only tail. 
        """
        setlevel = min(setlevel, levelmax)
        for l in range(0, setlevel+1):
            step = 2**l
            for i in range(T // step):
                sets.append([i*step, (i+1)*step-1])
    elif set_policy == "large":   
        for l in range(setlevel, levelmax+1):
            step = 2**l
            for i in range(T // step):
                sets.append([i*step, (i+1)*step-1])
    else : # fixed
        assert set_policy == "fixed", "set_policy must be one of ['small','large','fixed']"
        if setlevel <= levelmax:
            step = 2**setlevel
            for i in range(T // step):
                sets.append([i*step, (i+1)*step-1])
    # create causal mask
    t_idx = torch.arange(T, device=device)  # (T,)
    r_idx = torch.tensor([r for _,r in sets], device=device)  # (nset,)
    mask = t_idx.unsqueeze(-1) >= r_idx.unsqueeze(0)  # (T,nset)
    return sets, setlevel, levelmax, mask

def visualize_attention_matrix(
    att_matrix: torch.Tensor,
    batch_index: int = 0,
    num_batches: int = 4,
    save_path: Optional[str] = None,
    layer_index = None,
):
    """
    Visualize a 4D attention matrix (B, H, Q, K) for multiple batches and heads.

    Args:
        att_matrix: Attention weights or logits with shape (batch, heads, query, key).
        batch_index: Starting batch index to visualize (up to ``num_batches`` batches).
        num_batches: Maximum number of batches to show (defaults to 4 or whatever is available).
        title: Custom title for the plot. Defaults to an auto generated one.
        save_path: Optional filesystem path to save the figure.
        show: Whether to immediately display the figure via matplotlib.

    Returns:
        (figure, axis) tuple from matplotlib for further customization.
    """
    if att_matrix.ndim != 4:
        raise ValueError(f"Expected attention matrix with 4 dims (B, H, Q, K), got {att_matrix.shape}")
    if not (0 <= batch_index < att_matrix.size(0)):
        raise IndexError(f"batch_index {batch_index} out of range for batch size {att_matrix.size(0)}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "visualize_attention_matrix requires matplotlib to be installed."
        ) from exc

    total_batches = att_matrix.size(0)
    num_heads = att_matrix.size(1)
    batches_to_show = min(num_batches, total_batches - batch_index)
    if batches_to_show <= 0:
        raise ValueError("No batches available to visualize from the provided batch_index.")

    fig_width = 3 * num_heads
    fig_height =  3 * batches_to_show
    fig, axes = plt.subplots(batches_to_show, num_heads, figsize=(fig_width, fig_height))

    if batches_to_show == 1 and num_heads == 1:
        axes = [[axes]]
    elif batches_to_show == 1:
        axes = [axes]
    elif num_heads == 1:
        axes = [[ax] for ax in axes]

    colorbar_ref = None
    for row_idx, b_idx in enumerate(range(batch_index, batch_index + batches_to_show)):
        for head_idx in range(num_heads):
            att_slice = att_matrix[b_idx, head_idx]
            att_np = att_slice.detach().float().cpu().numpy()
            ax = axes[row_idx][head_idx]
            im = ax.imshow(att_np, aspect="auto", interpolation="nearest", cmap="viridis")
            if colorbar_ref is None:
                colorbar_ref = im

            if row_idx == batches_to_show - 1:
                ax.set_xlabel("Key index")
            else:
                ax.set_xticklabels([])
            if head_idx == 0:
                ax.set_ylabel(f"Batch {b_idx}\nQuery index")
            else:
                ax.set_yticklabels([])
            if row_idx == 0:
                ax.set_title(f"Head {head_idx}")

    if layer_index is not None:
        title = f"Attention weights in layer{layer_index}"
    else:
        title = "Attention weights"
    fig.suptitle(title)

    all_axes = [axis for row in axes for axis in row]
    # if colorbar_ref is not None:
    #     fig.colorbar(colorbar_ref, ax=all_axes, fraction=0.02, pad=0.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=300)
    import ipdb; ipdb.set_trace()
    return fig, axes


class CustomLinearAttention(LinearAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def preprocess(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        if self.num_kv_groups > 1:
            k = repeat(k, '... (h d) -> ... (h g) d', d=self.head_k_dim, g=self.num_kv_groups)
            v = repeat(v, '... (h d) -> ... (h g) d', d=self.head_v_dim, g=self.num_kv_groups)
        else:
            k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
            v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        q = self.feature_map_q(q)
        k = self.feature_map_k(k)

        if self.norm_q:
            q = q / (q.sum(-1, True) + 1e-4)
        if self.norm_k:
            k = k / (k.sum(-1, True) + 1e-4)
        return q, k, v

    def customforward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        mode = self.mode
        # 调用相应的算子
        if mode == 'chunk':
            assert cu_seqlens is None, "cu_seqlens should be None for chunk mode"
            o, final_state = chunk_linear_attn(
                q=q,
                k=k,
                v=v,
                normalize=self.do_feature_map_norm,
                output_final_state=True,
            )
        elif mode == 'fused_chunk':
            o, final_state = fused_chunk_linear_attn(
                q=q,
                k=k,
                v=v,
                normalize=self.do_feature_map_norm,
                output_final_state=True,
                cu_seqlens=cu_seqlens
            )
        elif mode == 'fused_recurrent':
            o, final_state = fused_recurrent_linear_attn(
                q=q,
                k=k,
                v=v,
                normalize=self.do_feature_map_norm,
                output_final_state=True,
                cu_seqlens=cu_seqlens
            )
        else:
            raise NotImplementedError
            
        o = self.norm(o)
        o = rearrange(o, '... h d -> ... (h d)')
        o = self.o_proj(o)
        
        return o, final_state

class SetAttention_Linear_Slow(nn.Module):
    def __init__(self, config, layer_index):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.level = config.attn.level
        self.levelrand = config.attn.levelrand
        self.k_mapping = config.attn.k_mapping
        self.v_mapping = config.attn.v_mapping
        self.set_policy = config.attn.set_policy
        self.feature_map = config.attn.feature_map
        if self.k_mapping:
            self.k_proj = nn.Linear((config.n_embd // config.n_head)**2, config.n_embd // config.n_head)
        if self.v_mapping:
            self.v_proj = nn.Linear((config.n_embd // config.n_head)**2, config.n_embd // config.n_head)
        # FLA linear attention
        self.linear_attn = CustomLinearAttention(
            mode = 'fused_recurrent',
            hidden_size=config.n_embd,  
            num_heads=self.n_head,
            feature_map=self.feature_map,
        )
        self.layer_index = layer_index

    def forward(self, x):
        
        B, T, C = x.size()
        nh = self.n_head
        hs = C // nh
        q, k, v = self.linear_attn.preprocess(x)  # q,k,v: (B, T, nh, hs)
        sets, setlevel, levelmax, mask = get_sets(T, self.levelrand, self.level, self.set_policy, x.device)
        nsets = len(sets)
        set_features = []
        K_mean = []
        V_mean = []
        
        if nsets > 0:
            for i in range(nsets):
                # print("sets",i)
                l_idx, r_idx = sets[i]
                q_slice = q[:, l_idx:r_idx+1, :]  # (B, len, nh, hs)
                k_slice = k[:, l_idx:r_idx+1, :]
                v_slice = v[:, l_idx:r_idx+1, :]
                K_mean.append(k_slice.mean(dim=1))  # (B, nh, hs)
                V_mean.append(v_slice.mean(dim=1))
                _, feature = self.linear_attn.customforward(q_slice, k_slice, v_slice)  #feature:(B,nh,hs,hs)
                set_features.append(feature)
            
            set_features = torch.stack(set_features, dim=1)  # (B, nsets, nh, hs, hs)
            set_features = set_features.view(B, nsets, nh, hs*hs) # (B, nset, nh, hs*hs)
            
            K_mean = torch.stack(K_mean, dim=1)  # (B, nset, nh, hs)
            V_mean = torch.stack(V_mean, dim=1)  # (B, nset, nh, hs)
            K = self.k_proj(set_features) if self.k_mapping else K_mean # (B, nset, nh, hs)
            V = self.v_proj(set_features) if self.v_mapping else V_mean # (B, nset, nh, hs)
            # import ipdb; ipdb.set_trace()
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,nset)
            # compute logits: q: (B,T,nh,hs) K: (B,nset,nh,hs) -> (B,nh,T,nset)
            att_logits = torch.matmul(q.transpose(1,2), K.transpose(1,2).transpose(-2,-1)) / (hs ** 0.5)  # (B,nh,T,nset)
            att_logits = att_logits.masked_fill(~mask, float('-inf'))
        else:
            # for inference. First few steps may not have any sets.
            att_logits = torch.zeros(B, nh, T, 0, device=x.device, dtype=x.dtype)  # (B,nh,T,0)   
            V = torch.zeros(B, 0, nh, hs, device=x.device, dtype=x.dtype)  # (B,0,nh,hs)
        #process tail
        if setlevel == 0 or self.smaller_sets:
            # no tail
            att = F.softmax(att_logits, dim=-1)  # (B,nh,T,nset)
            att = F.dropout(att, p=self.dropout, training=self.training)
            V = V.transpose(1,2)  # (B,nh,nset,hs)
            out = att @ V  # (B,nh,T,hs)
            out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
            out = self.resid_dropout(self.c_proj(out))
            return out

        # process tail
        Lmin = 2**setlevel
        tail_features = []
        K_mean = []
        V_mean = []
        for t in range(T):
            # print("tail",t)
            tail_len = t % (Lmin) + 1
            l_tail = t - tail_len + 1
            r_tail = t
            q_slice = q[:, l_tail:r_tail+1, :, :]  # (B, len, nh, hs)
            k_slice = k[:, l_tail:r_tail+1, :, :]  # (B, len, nh, hs)
            v_slice = v[:, l_tail:r_tail+1, :, :]  # (B, len, nh, hs)
            K_mean.append(k_slice.mean(dim=1))  # (B, nh, hs)
            V_mean.append(v_slice.mean(dim=1))
            _, feature = self.linear_attn.customforward(q_slice, k_slice, v_slice)  # (B, nh, hs, hs)
            tail_features.append(feature)
        K_mean = torch.stack(K_mean, dim=1)  # (B, T, nh, hs)
        V_mean = torch.stack(V_mean, dim=1)  # (B, T, nh, hs)
        tail_features = torch.stack(tail_features, dim=1)  # (B, T, nh, hs, hs)
        tail_features = tail_features.view(B, T, nh, hs*hs)  # (B, T, nh, hs*hs)
        K_tail = self.k_proj(tail_features) if self.k_mapping else K_mean  # (B, T, nh, hs)
        V_tail = self.v_proj(tail_features) if self.v_mapping else V_mean  # (B, T, nh, hs)

        attn_tail_logits = torch.sum(q * K_tail, dim=-1) / (hs**0.5)  # (B, T, nh)
        # concatenate
        att_logits = torch.cat([att_logits, attn_tail_logits.transpose(1, 2).unsqueeze(-1)], dim=-1)  # (B,nh,T,nset+1)
        att = F.softmax(att_logits, dim=-1)  # (B,nh,T,nset+1)
        att = self.attn_dropout(att)
        out = att[...,:-1] @ V.transpose(1,2)  # (B,nh,T,hs)
        out_tail = att[...,-1:] * V_tail.transpose(1,2)  # (B,nh,T,hs)
        out = out + out_tail  # (B,nh,T,hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out


"""
Faster Implementation using cu_seqlens
correctness verified against above implementation, 20x faster than above
TODO:can be further optimized by fusing two forward
""" 
class SetAttention_Linear(nn.Module):
    def __init__(self, config, layer_index):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.level = config.attn.level
        self.levelrand = config.attn.levelrand
        self.k_mapping = config.attn.k_mapping
        self.v_mapping = config.attn.v_mapping
        self.set_policy = config.attn.set_policy
        self.feature_map = config.attn.feature_map
        self.levelmax = config.attn.levelmax
        assert self.k_mapping == True and self.v_mapping == True, "k_mapping and v_mapping must be True"
        if self.k_mapping:
            self.k_proj = nn.Linear((config.n_embd // config.n_head)**2, config.n_embd // config.n_head)
        if self.v_mapping:
            self.v_proj = nn.Linear((config.n_embd // config.n_head)**2, config.n_embd // config.n_head)
        # FLA linear attention
        self.linear_attn = CustomLinearAttention(
            mode = 'fused_recurrent',
            hidden_size=config.n_embd,  
            num_heads=self.n_head,
            feature_map=self.feature_map,
        )
        self.layer_index = layer_index

    def forward(self, x, visualize = False):
        B, T, C = x.size()
        nh = self.n_head
        hs = C // nh
        q, k, v = self.linear_attn.preprocess(x)  # q,k,v: (B, T, nh, hs)
        # calc cumsum for fast mean computation
        k_cumsum = torch.cat([torch.zeros(B, 1, nh, hs, device=k.device), k.cumsum(dim=1)], dim=1)
        v_cumsum = torch.cat([torch.zeros(B, 1, nh, hs, device=k.device), v.cumsum(dim=1)], dim=1)

        sets, setlevel, levelmax, mask = get_sets(T, self.levelrand, self.level, self.levelmax, self.set_policy, x.device)
        # import ipdb; ipdb.set_trace()
        nsets = len(sets)
        set_features = []
        K_mean = []
        V_mean = []
        lens = []
        if len(sets) > 0:
            if self.set_policy == "small":
                
                for l in range(0, setlevel+1):
                    curlen = 2**l
                    tail = T % curlen
                    num_sets = (T - tail) // curlen
                    lens.extend([curlen]*num_sets)
                    # tried cumsum,find slower.
                    k_ = k[:, :T-tail, :, :].reshape(B,-1,curlen,nh,hs).mean(dim=2)  # (B, num_sets, nh, hs)
                    v_ = v[:, :T-tail, :, :].reshape(B,-1,curlen,nh,hs).mean(dim=2)  # (B, num_sets, nh, hs)
                    K_mean.append(k_)
                    V_mean.append(v_)
                idx = torch.cat([
                    torch.arange(0, T - (T % (2**l)))
                    for l in range(0, setlevel+1)
                ]).to(q.device)
                
            elif self.set_policy == "large":
                for l in range(setlevel, levelmax+1):
                    curlen = 2**l
                    tail = T % curlen
                    num_sets = (T - tail) // curlen
                    lens.extend([curlen]*num_sets)
                    # tried cumsum,find slower.
                    k_ = k[:, :T-tail, :, :].reshape(B,-1,curlen,nh,hs).mean(dim=2)  # (B, num_sets, nh, hs)
                    v_ = v[:, :T-tail, :, :].reshape(B,-1,curlen,nh,hs).mean(dim=2)  # (B, num_sets, nh, hs)
                    K_mean.append(k_)
                    V_mean.append(v_)
                idx = torch.cat([
                    torch.arange(0, T - (T % (2**l)))
                    for l in range(setlevel, levelmax+1)
                ]).to(q.device)
                
            else : # set_policy == fixed
                if setlevel <= levelmax:
                    curlen = 2**setlevel
                    tail = T % curlen
                    num_sets = (T - tail) // curlen
                    lens.extend([curlen]*num_sets)
                    k_ = k[:, :T-tail, :, :].reshape(B,-1,curlen,nh,hs).mean(dim=2)  # (B, num_sets, nh, hs)
                    v_ = v[:, :T-tail, :, :].reshape(B,-1,curlen,nh,hs).mean(dim=2)  # (B, num_sets, nh, hs)
                    K_mean.append(k_)
                    V_mean.append(v_)
                    idx = torch.arange(0, T - (T % (2**setlevel))).to(q.device)
                    
            q_all = q[:,idx, :, :].reshape(1,-1,nh,hs) # (1, :, nh, hs)
            k_all = k[:,idx, :, :].reshape(1,-1,nh,hs) # (1, :, nh, hs)
            v_all = v[:,idx, :, :].reshape(1,-1,nh,hs) # (1, :, nh, hs)
            lens = lens * B
            cu_seqlens = torch.tensor(lens, dtype=torch.long, device=x.device).cumsum(dim=0)
            cu_seqlens = torch.cat([torch.zeros((1,), dtype=torch.long, device=x.device), cu_seqlens], dim=0)
            _, feature = self.linear_attn.customforward(q_all, k_all, v_all, cu_seqlens=cu_seqlens) # feature:(B*nsets,nh,hs,hs)
            
            set_features = feature.view(B, nsets, nh, hs*hs) # (B, nset, nh, hs*hs)
            
            K_mean = torch.cat(K_mean, dim=1)  # (B, nset, nh, hs)
            V_mean = torch.cat(V_mean, dim=1)  # (B, nset, nh, hs)

            K = self.k_proj(set_features) if self.k_mapping else K_mean # (B, nset, nh, hs)
            V = self.v_proj(set_features) if self.v_mapping else V_mean # (B, nset, nh, hs)

            mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,nset)
            # compute logits: q: (B,T,nh,hs) K: (B,nset,nh,hs) -> (B,nh,T,nset)
            att_logits = torch.matmul(q.transpose(1,2), K.transpose(1,2).transpose(-2,-1)) / (hs ** 0.5)  # (B,nh,T,nset)
            att_logits = att_logits.masked_fill(~mask, float('-inf'))
        else:
            # for inference. First few steps may not have any sets.
            att_logits = torch.zeros(B, nh, T, 0, device=x.device, dtype=x.dtype)  # (B,nh,T,0)   
            V = torch.zeros(B, 0, nh, hs, device=x.device, dtype=x.dtype)  # (B,0,nh,hs)
        #process tail
        if setlevel == 0 or self.set_policy == "small":
            # no tail
            att = F.softmax(att_logits, dim=-1)  # (B,nh,T,nset)
            if visualize:
                print("visualizing attention matrix... T={}, nset={}, level={}".format(T, att_logits.shape[-1], setlevel))
                visualize_attention_matrix(att, batch_index=0, save_path="./out-img/attn.png",layer_index=self.layer_index)
            att = F.dropout(att, p=self.dropout, training=self.training)
            V = V.transpose(1,2)  # (B,nh,nset,hs)
            out = att @ V  # (B,nh,T,hs)
            out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
            out = self.resid_dropout(self.c_proj(out))
            return out

        # process tail, Larger sets only.
        """
        levelmax refers to the max level during training. During inference, 
        if setlevel > levelmax, there will be no sets, only tail.
        As there are no sets, tail includes all tokens. So Lmin = T+1
        """
        # import ipdb; ipdb.set_trace()
        if setlevel <= levelmax:
            Lmin = 2**setlevel 
        else :
            assert nsets == 0, "if setlevel > levelmax, there should be no sets"
            Lmin = T + 1
        tail_features = []
        K_mean = []
        V_mean = []
        t_range = torch.arange(T, device=x.device)  # (T,)
        tail_lens = t_range % (Lmin) + 1
        """
        This is a N^2 implementation, can be further optimized.
        [TODO]
        """
        idx = torch.cat([
            torch.arange(t - (t % Lmin), t + 1)
            for t in range(T)
        ]).to(q.device)
        q_all = q[:, idx, :, :].reshape(1,-1,nh,hs) # (1, :, nh, hs)
        k_all = k[:, idx, :, :].reshape(1,-1,nh,hs) # (1, :, nh, hs)
        v_all = v[:, idx, :, :].reshape(1,-1,nh,hs) # (1, :, nh, hs)
        lens = tail_lens.tolist() * B
        cu_seqlens = torch.tensor(lens, dtype=torch.long, device=x.device).cumsum(dim=0)
        cu_seqlens = torch.cat([torch.zeros((1,), dtype=torch.long, device=x.device), cu_seqlens], dim=0)
        _, feature = self.linear_attn.customforward(q_all, k_all, v_all, cu_seqlens=cu_seqlens)  # (B, nh, hs, hs)
        tail_features = feature.view(B, T, nh, hs*hs)
        
        l_tail = t_range - tail_lens + 1
        r_tail = t_range
        k_sum = k_cumsum[:, r_tail+1, :, :] - k_cumsum[:, l_tail, :, :]  # (B, T, nh, hs)
        v_sum = v_cumsum[:, r_tail+1, :, :] - v_cumsum[:, l_tail, :, :]  # (B, T, nh, hs)
        K_mean = k_sum / tail_lens.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (B, T, nh, hs)
        V_mean = v_sum / tail_lens.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (B, T, nh, hs)

        K_tail = self.k_proj(tail_features) if self.k_mapping else K_mean  # (B, T, nh, hs)
        V_tail = self.v_proj(tail_features) if self.v_mapping else V_mean  # (B, T, nh, hs)

        attn_tail_logits = torch.sum(q * K_tail, dim=-1) / (hs**0.5)  # (B, T, nh)
        # concatenate
        att_logits = torch.cat([att_logits, attn_tail_logits.transpose(1, 2).unsqueeze(-1)], dim=-1)  # (B,nh,T,nset+1)
        att = F.softmax(att_logits, dim=-1)  # (B,nh,T,nset+1)
        if visualize:
            print("visualizing attention matrix... T={}, nset={}, level={}".format(T, att_logits.shape[-1]-1, setlevel))
            visualize_attention_matrix(att, batch_index=0, save_path="./out-img/attn.png",layer_index=self.layer_index)
        att = self.attn_dropout(att)
        out = att[...,:-1] @ V.transpose(1,2)  # (B,nh,T,hs)
        out_tail = att[...,-1:] * V_tail.transpose(1,2)  # (B,nh,T,hs)
        out = out + out_tail  # (B,nh,T,hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out
