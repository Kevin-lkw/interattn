# -*- coding: utf-8 -*-
"""
Unfinished. Do not found cu_seqlen method

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple

from fla.layers.mamba import Mamba
from fla.layers.mamba2 import Mamba2

from setattn.setattn_legacy import get_sets

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

class SetAttention_Linear(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.level = config.level
        self.levelrand = config.levelrand
        self.k_mapping = config.k_mapping
        self.v_mapping = config.v_mapping
        self.smaller_sets = config.smaller_sets
        self.feature_map = config.feature_map
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

    def forward(self, x):
        
        B, T, C = x.size()
        nh = self.n_head
        hs = C // nh
        q, k, v = self.linear_attn.preprocess(x)  # q,k,v: (B, T, nh, hs)
        sets, setlevel, levelmax, mask = get_sets(T, self.levelrand, self.level, self.smaller_sets, x.device)
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
class SetAttention_Linear_Fast(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.level = config.level
        self.levelrand = config.levelrand
        self.k_mapping = config.k_mapping
        self.v_mapping = config.v_mapping
        self.smaller_sets = config.smaller_sets
        self.feature_map = config.feature_map
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

    def forward(self,x):
        B, T, C = x.size()
        nh = self.n_head
        hs = C // nh
        q, k, v = self.linear_attn.preprocess(x)  # q,k,v: (B, T, nh, hs)
        # calc cumsum for fast mean computation
        k_cumsum = torch.cat([torch.zeros(B, 1, nh, hs, device=k.device), k.cumsum(dim=1)], dim=1)
        v_cumsum = torch.cat([torch.zeros(B, 1, nh, hs, device=k.device), v.cumsum(dim=1)], dim=1)

        sets, setlevel, levelmax, mask = get_sets(T, self.levelrand, self.level, self.smaller_sets, x.device)
        nsets = len(sets)
        set_features = []
        K_mean = []
        V_mean = []
        lens = []
        if len(sets) > 0:
            # support larger sets first
            assert self.smaller_sets==False, "smaller_sets=False is required for fast version"
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
        t_range = torch.arange(T, device=x.device)  # (T,)
        tail_lens = t_range % (Lmin) + 1
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
        att = self.attn_dropout(att)
        out = att[...,:-1] @ V.transpose(1,2)  # (B,nh,T,hs)
        out_tail = att[...,-1:] * V_tail.transpose(1,2)  # (B,nh,T,hs)
        out = out + out_tail  # (B,nh,T,hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out