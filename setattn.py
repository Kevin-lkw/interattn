import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fla.layers import LinearAttention
class SetAttention_Linear(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
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
    def phi(self, x):
        if self.feature_map == 'identity':
            return x
        elif self.feature_map == 'relu':
            return F.relu(x)
        elif self.feature_map == 'elu_plus':
            return F.elu(x) + 1  # 保证正值
        elif self.feature_map == 'softplus':
            return F.softplus(x)
        elif self.feature_map == 'exp':
            return torch.exp(x)
        elif self.feature_map == 'sigmoid':
            return torch.sigmoid(x)
        elif self.feature_map == 'swish':
            return x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported feature_map: {self.feature_map}")


    def get_slice(self, x, l_idx, r_idx):
        r = x[:, :, r_idx, :]  # (B, nh, nset, hs)
        l = x[:, :, l_idx-1, :] 
        mask_shape = [1, 1, len(l_idx)] + [1]*(x.dim()-3)
        mask = (l_idx.view(*mask_shape) >= 1)
        l = l * mask.to(x.device, x.dtype)
        return r - l
    
    def forward(self, x):
        B, T, C = x.size()
        nh = self.n_head
        hs = C // nh
        
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, nh, hs).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        q_phi = self.phi(q)  # (B, nh, T, hs)
        k_phi = self.phi(k)  # (B, nh, T, hs)
        if self.k_mapping or self.v_mapping:
            kv = k_phi.unsqueeze(-1) * v.unsqueeze(-2)  # outer product: (B, nh, T, hs, hs)
            kv = kv.view(B, nh, T, hs*hs)
            if self.k_mapping:
                k_phi = self.k_proj(kv).view(B, nh, T, hs)  # (B, nh, T, hs)
            if self.v_mapping:
                v = self.v_proj(kv).view(B, nh, T, hs)  # (B, nh, T, hs)
        k_cum = torch.cumsum(k_phi, dim=2)  # (B, nh, T, hs)
        v_cum = torch.cumsum(v, dim=2)  # (B, nh, T, hs)

        sets = []
        levelmax = math.floor(math.log2(T))
        setlevel = torch.randint(0,levelmax+1,()) if self.levelrand else self.level
        if self.smaller_sets:
            for l in range(0, setlevel+1):
                step = 2**l
                for i in range(0, T // step, 2):
                    sets.append([i*step, (i+1)*step-1])
        else:   
            for l in range(setlevel, levelmax+1):
                step = 2**l
                for i in range(T // step):
                    sets.append([i*step, (i+1)*step-1])
        nsets = len(sets)
        t_idx = torch.arange(T, device=x.device)  # (T,)

        if nsets > 0:
            # construct K V and mask for sets
            bounds = torch.tensor(sets, device=x.device)  # (nset,2)
            l_idx, r_idx = bounds[:,0], bounds[:,1]
            K = self.get_slice(k_cum, l_idx, r_idx)  # (B, nh, nset, hs)
            V = self.get_slice(v_cum, l_idx, r_idx)  # (B, nh, nset, hs, hs)
            # V = torch.randn(B, nh, nsets, hs, device=x.device, dtype=x.dtype)
            # generate mask according to bounds
            mask = t_idx.unsqueeze(-1) >= r_idx.unsqueeze(0)  # (T,nset)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,nset)
            # compute logits
            # q: (B,nh,T,hs) K: (B,nh,nset,hs) -> (B,nh,T,nset)
            att_logits = torch.matmul(q, K.transpose(-2, -1)) / (hs ** 0.5)  # (B,nh,T,nset)
            att_logits = att_logits.masked_fill(~mask, float('-inf'))

        else :
            # for inference. First few steps may not have any sets.
            att_logits = torch.zeros(B, nh, T, 0, device=x.device, dtype=x.dtype)  # (B,nh,T,0)   
            V = torch.zeros(B, nh, 0, hs, device=x.device, dtype=x.dtype)  # (B,nh,0,hs) 

        if setlevel == 0 or self.smaller_sets:
            # no tail
            att = F.softmax(att_logits, dim=-1)  # (B,nh,T,nset)
            att = self.attn_dropout(att)
            out = att @ V  # (B,nh,T,hs)
            out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
            out = self.resid_dropout(self.c_proj(out))
            return out
        # process tail
        Lmin = 2**setlevel
        tail_len = t_idx % (Lmin) + 1
        l_tail = t_idx - tail_len + 1
        r_tail = t_idx
        K_tail = self.get_slice(k_cum, l_tail, r_tail)  # (B, nh, T, hs)
        V_tail = self.get_slice(v_cum, l_tail, r_tail) # (B, nh, T, hs, hs)
        
        # q: (B,nh,T,hs) K_tail: (B,nh,T,hs) -> (B,nh,T)
        att_tail_logits = torch.sum(q_phi * K_tail, dim=-1) / (hs**0.5)

        # concatenate
        att_logits = torch.cat([att_logits, att_tail_logits.unsqueeze(-1)], dim=-1)  # (B,nh,T,nset+1)

        att = F.softmax(att_logits, dim=-1)  # (B,nh,T,nset+1)
        att = self.attn_dropout(att)

        # att (B,nh,T,nset) V(B,nh,nset,hs) -> (B,nh,T,hs)
        out = att[...,:-1] @ V  # (B,nh,T,hs)
        # att[...,-1] (B,nh,T) V_tail (B,nh,T,hs) -> (B,nh,T,hs)
        out_tail = att[...,-1:] * V_tail  # (B,nh,T,hs)
        out = out + out_tail  # (B,nh,T,hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out


class SetAttention_Linear_fla(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # 基础配置
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.level = config.level
        self.levelrand = config.levelrand
        self.k_mapping = config.k_mapping
        self.v_mapping = config.v_mapping
        self.smaller_sets = config.smaller_sets
        
        # FLA LinearAttention 配置
        self.mode = getattr(config, 'linear_attn_mode', 'fused_chunk')
        self.feature_map = getattr(config, 'feature_map', 'elu')
        self.expand_k = getattr(config, 'expand_k', 1.0)
        self.expand_v = getattr(config, 'expand_v', 1.0)
        
        # 创建 FLA LinearAttention 实例用于集合内聚合
        self.linear_attn = LinearAttention(
            mode=self.mode,
            hidden_size=config.n_embd // config.n_head,  # 每个头的维度
            expand_k=self.expand_k,
            expand_v=self.expand_v,
            num_heads=1,  # 每次处理一个头
            feature_map=self.feature_map,
            tie_feature_map_qk=True,
            output_norm='identity',  # 我们自己处理归一化
            norm_q=False,
            norm_k=False,
            do_feature_map_norm=getattr(config, 'do_feature_map_norm', False)
        )
    
    