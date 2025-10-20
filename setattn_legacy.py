"""
legacy Set Attention implementation
with a set division policy (uniform,DFS,BFS) and set aggregation method
set division can be different for every time step.
slow, not parallelable, thus decrpecated
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from collections import deque
class FastCausalSetAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # set
        self.set_number = config.set_number
        self.set_policy = build_set_policy(config.set_policy,self.set_number)
        self.set_aggr = build_set_aggr(config.set_aggr)
        # print("initialize Set Attention, setnumber = ",self.set_number, "," \
        #     ", set_poilcy = ",config.set_policy, ", set_aggr = ",config.set_aggr)
    def phi(self, x):
        return x
    def forward(self,x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        nh = self.n_head
        hs = C // self.n_head
        
        q_phi = self.phi(q)  # (B, nh, T, hs)
        k_phi = self.phi(k)  # (B, nh, T, hs)
        # Cumulative sums
        k_cum = torch.cumsum(k_phi, dim=2)  # (B, nh, T, hs)
        kv = k_phi.unsqueeze(-1) * v.unsqueeze(-2)  # outer product: (B, nh, T, hs, hs)
        kv_cum = torch.cumsum(kv, dim=2)  # (B, nh, T, hs, hs)

        sets = [[0,1],[0.5,1],[0,0.5],[0.75,1]]
        nset = len(sets)
        all_l_idx = []
        all_r_idx = []
        for l, r in sets:
            l_idx = torch.arange(T, device=k_cum.device) * l - 1
            r_idx = torch.floor(torch.arange(T, device=k_cum.device) * r).long()
            l_idx = l_idx.clamp(min=-1).long()   # -1 表示要 special case
            all_l_idx.append(l_idx)  # (T,)
            all_r_idx.append(r_idx)  # (T,)
        
        # stack -> (nset, T)
        all_l_idx = torch.stack(all_l_idx)  
        all_r_idx = torch.stack(all_r_idx) 
        # 2. broadcast 成 (B, nh, nset, T, hs)
        def gather_prefix(prefix, idx):
            # prefix: (B, nh, T, *feat)  where *feat can be (hs) or (hs,hs)
            # idx: (nset, T)

            B, nh, T = prefix.shape[:3]
            nset = idx.size(0)
            feat_shape = prefix.shape[3:]   # could be (hs,) or (hs,hs)

            # expand idx to match feature shape
            idx = idx.view(1, 1, nset, T, *([1]*len(feat_shape))) \
                    .expand(B, nh, nset, T, *feat_shape)

            # expand prefix for nset
            prefix = prefix.unsqueeze(2).expand(B, nh, nset, T, *feat_shape)

            # gather on the time dimension (dim=3)
            gathered = torch.gather(prefix, 3, idx)

            return gathered  # (B, nh, nset, T, *feat_shape)

        K_r = gather_prefix(k_cum, all_r_idx)
        K_l = gather_prefix(k_cum, all_l_idx.clamp(min=0))  # -1 先用 0 替代
        K_l = K_l * (all_l_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1) >= 0)  # mask掉 -1

        KV_r = gather_prefix(kv_cum, all_r_idx)  # (B,nh,lenset,T,hs,hs)
        KV_l = gather_prefix(kv_cum, all_l_idx.clamp(min=0))
        KV_l = KV_l * (all_l_idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) >= 0)
        # 3. 区间差分
        K_hat = (K_r - K_l).permute(0,1,3,2,4)   # (B, nh, T, nset, hs)
        V_hat = (KV_r - KV_l).permute(0,1,3,2,4,5)  # (B, nh, T, nset, hs, hs)

        # 1. q 与 K_hat 点积，得到注意力 logits
        # q: (B,nh,T,hs)
        # K_hat: (B,nh,T,nset,hs)
        att_logits = torch.einsum("bthd,bthkd->bthk", q, K_hat)  # (B,nh,T,nset)

        # 2. softmax
        att = F.softmax(att_logits, dim=-1)  # (B,nh,T,nset)
        att = self.attn_dropout(att)
        # 3. 注意力加权 V_hat
        # V_hat: (B,nh,T,nset,hs,hs)
        # 先把 att broadcast 成 (B,nh,T,nset,1,1)
        att_exp = att.unsqueeze(-1).unsqueeze(-1)

        # 加权求和 -> (B,nh,T,hs,hs)
        V_summary = (att_exp * V_hat).sum(dim=3)

        # 4. 最终输出: q_t @ V_summary
        # (B,nh,T,1,hs) @ (B,nh,T,hs,hs) -> (B,nh,T,1,hs)
        out = torch.matmul(q.unsqueeze(-2), V_summary)  # (B,nh,T,1,hs)
        out = out.squeeze(-2)  # (B,nh,T,hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out
            
class CausalSetAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # set
        self.set_number = config.set_number
        self.set_policy = build_set_policy(config.set_policy,self.set_number)
        self.set_aggr = build_set_aggr(config.set_aggr)
        # print("initialize Set Attention, setnumber = ",self.set_number, "," \
        #     ", set_poilcy = ",config.set_policy, ", set_aggr = ",config.set_aggr)
    def forward(self,x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # implement set attnetion
        output = torch.zeros_like(q)  # (B, nh, T, hs)
        import time;
        # t0=time.time()
        self.set_aggr.reset()
        for t in range(T):
            k_t = k[:, :, t, :]  # (B, nh, hs)
            v_t = v[:, :, t, :]  # (B, nh, hs)
            q_t = q[:, :, t, :]  # (B, nh, hs)

            # Divide [0, t] into `set_number` groups using the set policy
            sets = self.set_policy.divide(t)  # List[List[int]]
            self.set_aggr.insert(k_t,v_t)
            keys,vals = self.set_aggr(sets,q_t) # List of tensor (B,nh,hs)

            # Stack aggregated K and V: (B, nh, num_sets, hs)
            # print("len = ",len(keys),"shape = ",keys[0].shape)

            K = torch.stack(keys, dim=2)
            V = torch.stack(vals, dim=2)

            # Scaled dot-product attention
            q_t_exp = q_t.unsqueeze(2)  # (B, nh, 1, hs)
            att = torch.matmul(q_t_exp, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)  # (B, nh, 1, num_sets)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            out_t = torch.matmul(att, V)  # (B, nh, 1, hs)
            output[:, :, t, :] = out_t.squeeze(2)  # Store result

        # Merge heads and project output
        out = output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        t1=time.time()
        # print("time",t1-t0)
        return out

def build_set_policy( policy_name: str, set_number: int):
    if policy_name == "uniform":
        return UniformSetPolicy(set_number)
    elif policy_name == "bfs":
        return BFSPolicy(set_number)
    elif policy_name == "dfs":
        return DFSPolicy(set_number)
    else:
        raise ValueError(f"Unknown set division policy: {policy_name}")
    
class SetDivisionPolicy:
    def __init__(self, set_number: int):
        self.set_number = set_number
    
    def divide(self, t: int) -> List[List[int]]:
        # dividing [0,t] into set_number sets.
        raise NotImplementedError

class UniformSetPolicy(SetDivisionPolicy):
    def divide(self, t: int) -> List[List[int]]:
        chunk_size = math.ceil(t / self.set_number)
        chunks = [[i,min(i+chunk_size, t)] for i in range(0, t, chunk_size)]
        return chunks

class BFSPolicy(SetDivisionPolicy):
    def divide(self, t: int) -> List[List[int]]:
        result = []
        queue = deque()
        queue.append((0, t))  
        while queue:
            l, r = queue.popleft()
            result.append([l,r])
            if len(result) == self.set_number:
                break
            if l < r:
                m = (l + r) // 2
                queue.append((m+1, r))
                queue.append((l, m))
        return result

class DFSPolicy(SetDivisionPolicy):
    def divide(self, t: int) -> List[List[int]]:
        return self.recursive_divide(0,t,self.set_number)
    
    def recursive_divide(self,l,r,set_num) -> List[List[int]]:
        res = [[l,r]]
        if l == r or set_num == 1:
            return res
        m = (l+r) // 2
        set_num = set_num - 1

        right_set = self.recursive_divide(m+1,r,set_num)
        set_num = set_num - len(right_set)
        res = res + right_set
        if set_num > 0:
            left_set = self.recursive_divide(l,m,set_num)
            res = res + left_set
        return res
    
"""
Set Aggregator part
"""
def build_set_aggr(aggr_name: str):
    if aggr_name == "linear":
        return LinearAttentionAggregator()
    elif aggr_name == "mamba":
        return MambaAggregator()
    else :
        raise ValueError(f"Unknown set aggregate policy: {aggr_name}")
class SetAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_cache = []
        self.v_cache = []
        self.embed_cache = [] # every element is a 

    def reset(self):
        """Clear cached keys and values"""
        self.k_cache.clear()
        self.v_cache.clear()
        self.embed_cache.clear()

    def insert(self, k: torch.Tensor, v: torch.Tensor):
        """
        Insert key and value at current time step.
        Args:
            k: (B, nh, hs)
            v: (B, nh, hs)
        """
        self.k_cache.append(k)  # list of (B, nh, hs)
        self.v_cache.append(v)

    def forward(self, index_list: List[List[int]], qt: torch.Tensor) -> Tuple[List[torch.Tensor],List[torch.Tensor]]:
        """
        Aggregate each group [l, r] into a single summary vector
        Args:
            index_list: List of [l, r] index pairs (inclusive)
        Returns:
            List of tensors of shape (B, nh, hs), one per [l, r]
        """
        raise NotImplementedError("Subclasses must implement this method")

class LinearAttentionAggregator(SetAggregator):
    def __init__(self):
        super().__init__()
    def forward(self, index_list: List[List[int]], qt: torch.tensor) -> Tuple[List[torch.Tensor],List[torch.Tensor]]:
        """
        Simulate linear attention using dot-product between fixed query and keys
        """
        B, nh, hs = self.k_cache[0].shape
        assert qt.shape == (B,nh,hs) , f"qt shape {qt.shape} != {(B, nh, hs)}"
        # return self.k_cache,self.v_cache
        rk = []
        rv = []
        cur = len(self.embed_cache)
        for index, (l, r) in enumerate(index_list):
            """
            q: (B,nh,hs), hs = hidden dimension
            k: (B,nh,hs)
            v: (B,nh,hs)
            (q dot k) = k^T q = (B,nh,1,hs),(B,nh,hs,1) -> (B,nh,1,1)  
            v(k^T q) =  (vk^T) q
            """
            if index >= cur:
                E = torch.zeros(B,nh,hs,hs,device=qt.device,dtype=qt.dtype)
                khat = torch.zeros(B,nh,hs,device=qt.device,dtype=qt.dtype)
                for i in range(l,r+1):
                    K = self.k_cache[i].unsqueeze(-1) # (B,nh,hs,1)
                    V = self.v_cache[i].unsqueeze(-1) # (B,nh,hs,1)
                    E = E + V @ K.transpose(-1,-2) # (B,nh,hs,hs) 
                    khat = khat + self.k_cache[i]
                self.embed_cache.append((E,khat,l,r))
            else :
                E,khat,l0,r0 = self.embed_cache[index]
                for i in range(l0+1,l+1):
                    K = self.k_cache[i].unsqueeze(-1) # (B,nh,hs,1)
                    V = self.v_cache[i].unsqueeze(-1) # (B,nh,hs,1)
                    E = E - V @ K.transpose(-1,-2) # (B,nh,hs,hs) 
                    khat = khat - self.k_cache[i]
                for i in range(r0+1,r+1):
                    K = self.k_cache[i].unsqueeze(-1) # (B,nh,hs,1)
                    V = self.v_cache[i].unsqueeze(-1) # (B,nh,hs,1)
                    E = E + V @ K.transpose(-1,-2) # (B,nh,hs,hs) 
                    khat = khat + self.k_cache[i]
                self.embed_cache[index] = (E,khat,l,r)
            
            E, khat, _, _ = self.embed_cache[index]
            vhat = (E @ qt.unsqueeze(-1)).squeeze(-1) # (B,nh,hs)
            khat = khat / (r-l+1)
            rk.append(khat)
            rv.append(vhat)  
        return rk, rv

class MambaAggregator(SetAggregator):
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    set_num = 7
    T = 10
    policy = build_set_policy("bfs",set_num)
    divition = policy.divide(T)
    print(divition)
    B = 1
    nh = 1
    hs = 1
    aggr = build_set_aggr("linear")
    for i in range(T+1):
        K = torch.randn(B,nh,hs)
        V = torch.randn(B,nh,hs)
        aggr.insert(K,V)
    k,v = aggr(divition,torch.randn(B,nh,hs))
    print(k,v)
