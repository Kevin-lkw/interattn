import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from collections import deque
"""
Set divition policy part
"""
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
