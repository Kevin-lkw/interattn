
from transformers import AutoModelForCausalLM
import torch
from torch.nn import functional as F
import os
from transformers import AutoTokenizer
import math
import numpy as np

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

llama_model = "meta-llama/Llama-2-7b-hf"
model_name = "llama-2-7b-hf"
dtype = torch.float32
device = "cuda:0"
# construct the last layer of llama
model = AutoModelForCausalLM.from_pretrained(
    llama_model,
    dtype=dtype,  
    device_map=device,
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(
    llama_model,
    use_fast=False,        
)
dataset_name = "wikitext"
start = 0
kv = torch.load(f"../{model_name}_{dataset_name}_st{start}.pt", weights_only=False)
model_config = kv["model_config"]
rope_qkv = kv["after_rope"]
inputs = kv["input"]
outputs = kv["output"]
attn_output = kv["attention_output"]
layer_input = kv["layer_input"]
gt_label = kv["gt_label"]
# print("model_config", model_config)
L = model_config.num_hidden_layers
print(model)


def get_attention_map_after_rope(layer_idx, causal=True, dtype=torch.bfloat16, device="cuda"):
    """
    返回: attn [seq_len, seq_len] (softmax 后)
    """
    Q = rope_qkv[layer_idx]['q']  # [B, nh, seq, hd]
    K = rope_qkv[layer_idx]['k']  # [B, nh, seq, hd]

    q = Q[0].to(dtype).to(device)  # [nh, seq, hd]
    k = K[0].to(dtype).to(device)  # [nh, seq, hd]
    
    hd = q.shape[-1]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)  # [nh, seq, seq]

    if causal:
        seq = scores.shape[1]
        mask = torch.triu(torch.ones(seq, seq, device=scores.device, dtype=torch.bool), diagonal=1).unsqueeze(0)  # [1, seq, seq]
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return scores, attn

def optimize_alpha_star(layer_idx, head_idx, pos_list ,training_steps, lr, mask, device="cuda"):
    """
    head: int, the head to optimize, int or list of int
    pos_list: list of int, the positions to optimize
    """
    if isinstance(head_idx, int):
        head_idx = [head_idx]
        
    # a[n_pos,seq]
    n_pos = len(pos_list)
    # a_param = torch.zeros(len(head_idx), n_pos, 4096, device=device, requires_grad=True)
    a_param = torch.nn.Parameter(torch.randn(len(head_idx), n_pos, 4096, device=device) * 0.1)
    # attention_score = get_attention_map_after_rope(layer_idx, causal=True, dtype=dtype, device=device)
    # a_param = attention_score[head_idx].to(device)[:, pos_list, :].clone().detach().requires_grad_(True)
    a_param.retain_grad()
    
    opt = torch.optim.Adam([a_param], lr=lr)
    
    # compute the constant part
    residual_attn_in = layer_input[layer_idx][0,pos_list].to(device) # [n_pos, hidden_size]
    original = attn_output[layer_idx]['output'][0,pos_list].permute(1,0,2).to(device) # [nh, n_pos, hd]
    V_head = rope_qkv[layer_idx]['v'].to(device)[0][head_idx]  # [nh, seq, hd]
    layer = model.model.layers[layer_idx]
    
    with torch.no_grad():
        # calculate gt distribution p*
        output = original.clone() # [nh, n_pos, hd]
        hidden = layer.self_attn.o_proj(output.permute(1,0,2).reshape(len(pos_list), -1)) # [n_pos, hidden_size]
        hidden = hidden + residual_attn_in # add residual
        # hidden = modelNorm(hidden)
        gt_logits = model.lm_head(hidden) # [n_pos, vocab_size]
        p_teacher = F.softmax(gt_logits.float(), dim=-1).detach()  # [n_pos, vocab_size]
        logp_teacher = F.log_softmax(gt_logits.float(), dim=-1).detach()  # [n_pos, vocab_size]
    losses = []
    p_alpha = 0
    for step in range(training_steps):
        alpha = F.softmax(a_param + mask, dim=-1)  
        V_new = alpha @ V_head.float() # alpha [nh, n_pos, seq] @ V_head [nh, seq, hd] -> [nh, n_pos, hd]
        V_new = V_new.to(original.dtype)
        output = original.clone()
        output[head_idx] = V_new.to(V_head.dtype)
        hidden = output.permute(1,0,2).reshape(len(pos_list), -1) # [n_pos, hidden_size]
        hidden = layer.self_attn.o_proj(hidden) # [n_pos, hidden_size]
        hidden = hidden + residual_attn_in # add residual
        # hidden = modelNorm(hidden)
        hidden = model.lm_head(hidden) # [n_pos, vocab_size]
        logits = hidden
        p_alpha = F.softmax(logits.float(), dim=-1) # [n_pos, vocab_size]
        logp_student = F.log_softmax(logits.float(), dim=-1)
        # KL-Divergence loss (CE loss)
        loss = (p_teacher * (logp_teacher-logp_student)).sum(dim=-1).mean()
        
        losses.append(loss.item())
        if step % 100 == 0:
            print("step", step, "loss:", loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    # verify KKT conditions
    alpha = F.softmax(a_param + mask, dim=-1)
    
    return alpha, p_alpha, p_teacher, losses


def gen_mask(layer_idx, pos_list, head_idx, strategy, budget, prompt_len=4032, seq_len=4096):
    """
    Return mask for alpha_param, with shape [nh, n_pos, seq_len]
    """
    mask = torch.zeros(len(pos_list), seq_len, device=device)

    visible = int(seq_len*budget)
    print(f"visible {visible} tokens for strategy {strategy} with budget {budget}")
    if strategy == "recency":
        for i, pos in enumerate(pos_list):
            if pos > visible:
                mask[i, :pos-visible] = float("-inf")
        mask = mask.unsqueeze(0).expand(len(head_idx), -1, -1)  # [nh, n_pos, seq_len]
    elif strategy == "random":
        for i,pos in enumerate(pos_list):
            if pos > visible:
                idx = torch.randperm(pos+1)[:visible]
                idx_list = idx.tolist()
                mask_list = list(set(range(pos+1)) - set(idx_list))
                mask[i, mask_list] = float("-inf")
        mask = mask.unsqueeze(0).expand(len(head_idx), -1, -1)  # [nh, n_pos, seq_len]
    elif strategy == "attention_topk":
        attention_unnormalize, _ = get_attention_map_after_rope(layer_idx, causal=True, dtype=dtype, device=device)
        # this is logits before softmax, but since we only care about topk, it's fine
        mask = torch.zeros(len(head_idx), len(pos_list), seq_len, device=device)
        for head in head_idx:
            attention_score_head = attention_unnormalize[head]  # [seq, seq]
            for i,pos in enumerate(pos_list):
                if pos > visible:
                    topk = torch.topk(attention_score_head[pos,:pos+1], k=visible, largest=True).indices
                    idx_list = topk.tolist()
                    mask_list = list(set(range(pos+1)) - set(idx_list))
                    mask[head, i, mask_list] = float("-inf")
                    
    elif strategy == "h2o":
        recent_budget = visible // 2
        hh_budget = visible - recent_budget  # handle odd visible

        _, attention_score = get_attention_map_after_rope(layer_idx, causal=True, dtype=dtype, device=device)
        
        mask = torch.zeros(len(head_idx), len(pos_list), seq_len, device=device)
        for out_h, head in enumerate(head_idx):
            attention_score_head = attention_score[head]  # [seq_len, seq_len]

            # accumulated_attention[j] = how much token j has been attended to so far
            accumulated_attention = torch.zeros(seq_len, device=device)

            # ------------------------------------------------------------
            # 1) PREFILL initialization:
            #    accumulate attention received from prompt queries [0, prompt_len)
            # ------------------------------------------------------------
            # Query q attends to keys [0..q] under causal mask, so summing rows
            # gives "received attention so far" for each key position.
            if prompt_len > 0:
                accumulated_attention += attention_score_head[:prompt_len, :].sum(dim=0)

            # ------------------------------------------------------------
            # 2) ONLINE decode:
            #    for each decode position pos:
            #      - select HH using accumulated score BEFORE current step
            #      - keep recent window
            #      - then update accumulated score with current query row
            # ------------------------------------------------------------
            for i, pos in enumerate(pos_list):
                total_available = pos + 1  # keys in [0, pos]

                # no need to evict if current context length <= visible
                if total_available <= visible:
                    # still update score for next step
                    accumulated_attention += attention_score_head[pos]
                    continue

                # recent window: last `recent_budget` positions in [0, pos]
                cur_recent_budget = min(recent_budget, total_available)
                recent_start = total_available - cur_recent_budget
                recent_idx = torch.arange(recent_start, total_available, device=device)

                # HH candidates exclude recent window to avoid overlap
                hh_candidate_end = recent_start  # candidates are [0, recent_start)
                cur_hh_budget = min(hh_budget, hh_candidate_end)

                keep = torch.zeros(total_available, dtype=torch.bool, device=device)

                # keep recent tokens
                keep[recent_idx] = True

                # keep heavy hitters from older prefix
                if cur_hh_budget > 0:
                    hh_scores = accumulated_attention[:hh_candidate_end]
                    topk_hh = torch.topk(hh_scores, k=cur_hh_budget, largest=True).indices
                    keep[topk_hh] = True

                # mask out everything else in [0, pos]
                mask[out_h, i, :total_available][~keep] = float("-inf")

                # IMPORTANT:
                # update AFTER building current-step mask,
                # because H2O uses only preceding attention statistics
                accumulated_attention += attention_score_head[pos]
            
                
    elif strategy == "kvmerger":
        ...
    else :
        raise ValueError(f"Unknown strategy {strategy}")            
    # causal mask
    for i, pos in enumerate(pos_list):
        mask[:, i, pos+1:] = float("-inf")
    return mask  # [n_heads, n_pos, 4096]

head_idx = list(range(32)) # [0,...,31]
pos_list = list(range(4096-64, 4096))
result = {}
strategy = "h2o"
layer_idx_list = [31]
for layer_idx in layer_idx_list:
    save_path = f"../result/layer{layer_idx}/{dataset_name}/{strategy}/result.pt"
    if os.path.exists(save_path):
        result = torch.load(save_path)
        print(f"Loaded existing results for layer {layer_idx}. Existing budgets: {list(result.keys())}")
    else:
        result = {}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"No existing file found for layer {layer_idx}, starting a new one.")
        
    for budget in [0.0005]:
        if budget in result:
            print(f"Budget {budget} already exists in layer {layer_idx}, skipping.")
            continue
        mask = gen_mask(layer_idx, pos_list, head_idx=head_idx, strategy=strategy, budget=budget)
        print(f"Optimizing alpha_star for layer {layer_idx}, budget {budget}")
        alpha, p_alpha, p_teacher, loss = optimize_alpha_star(
            layer_idx=layer_idx, head_idx=head_idx, pos_list=pos_list, training_steps=10000, lr=0.05, mask=mask, device=device)
        print( f"final loss for layer {layer_idx} with budget {budget}: ", loss[-1])
        result[budget] = (alpha, p_alpha, p_teacher, loss)

    # save the result
    torch.save(result, f"../result/layer{layer_idx}/{dataset_name}/{strategy}/result.pt")
    print("Optimization completed and results saved.")