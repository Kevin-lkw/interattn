import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
import os
from transformers import AutoTokenizer
import math

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
Wnorm = kv["Wnorm"] # Shape (hidden_size,)
Wlm = kv["Wlm"] # Shape (vocab_size, hidden_size)
inputs = kv["input"]
outputs = kv["output"]
attn = kv["last_layer_attention"]
last_layer_param  = kv["last_layer"]
last_layer_input = kv["last_layer_input"]
gt_label = kv["gt_label"]
# print("model_config", model_config)
L = model_config.num_hidden_layers



from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer

device = "cuda"
# constant part
modelNorm = LlamaRMSNorm(model_config.hidden_size, eps=model_config.rms_norm_eps).half().to(device)
modelNorm.load_state_dict({"weight": Wnorm.to(device)}, strict=True)
modelNorm.requires_grad_(False)
modelNorm.eval()

layer = LlamaDecoderLayer(model_config, layer_idx=L-1).half().to(device)
layer.load_state_dict(last_layer_param, strict=True)
layer.eval()
def plain_forward(alpha, head_idx, V_head, pos_list, original, residual_attn_in, Wo, Wlm):
        V_new = alpha.to(V_head.dtype) @ V_head # [n_pos, hd]
        output = original.clone()
        output[head_idx] = V_new
        hidden = output.permute(1,0,2).reshape(len(pos_list), -1) # [n_pos, hidden_size]
        hidden = hidden @ Wo.T # [n_pos, hidden_size]
        hidden = hidden + residual_attn_in # add residual
        
        hidden = modelNorm(hidden)
        hidden = hidden @ Wlm.T # [n_pos, vocab_size]
        return hidden 

def mlp_forward(alpha, head_idx, V_head, pos_list, original, residual_attn_in, Wo, Wlm):
    V_new = alpha.to(V_head.dtype) @ V_head # [n_pos, hd]
    output = original.clone()
    output[head_idx] = V_new
    hidden = output.permute(1,0,2).reshape(len(pos_list), -1) # [n_pos, hidden_size]
    hidden = hidden @ Wo.T # [n_pos, hidden_size]
    hidden = hidden + residual_attn_in # add residual
    residual = hidden
    hidden = layer.post_attention_layernorm(hidden)
    
    # MLP
    hidden = layer.mlp(hidden)
    hidden = hidden + residual
    
    
    hidden_states_normed = modelNorm(hidden)
    logits = hidden_states_normed @ Wlm.T
    return logits

    
def optimize_alpha_star(head_idx, pos_list ,training_steps=100, lr=0.5, device="cuda"):
    """
    head: int, the head to optimize
    pos_list: list of int, the positions to optimize
    """
    # a[n_pos,seq]
    n_pos = len(pos_list)
    # a_param = torch.zeros(n_pos, 4096, device=device, requires_grad=True)
    a_param = torch.randn(n_pos, 4096, device=device, requires_grad=True)
    
    mask = torch.zeros(n_pos, 4096, device=device)
    for i, pos in enumerate(pos_list):
        mask[i, pos+1:] = float("-inf")
    
    gt_y = gt_label[0, pos_list].to(device)
    
    opt = torch.optim.Adam([a_param], lr=lr)
    
    # compute the constant part
    residual_attn_in = last_layer_input['hidden_states'][0,pos_list].to(device) # [n_pos, hidden_size]
    original = attn['output'][0,pos_list].permute(1,0,2).to(device) # [nh, n_pos, hd]
    V_head = rope_qkv[L-1]['v'].to(device)[0][head_idx]  # [B, nh, seq, hd]
    Wo = last_layer_param['self_attn.o_proj.weight'].to(device)
    Wlm = kv["Wlm"].to(device)
    
    # def head_ablate_loss(head):
    #     output = original.clone()
    #     output[head].zero_()   # 或者 output[head] = 0
    #     hidden = output.permute(1,0,2).reshape(n_pos, -1)
    #     hidden = hidden @ Wo.T
    #     hidden = hidden + residual_attn_in
    #     residual = hidden
    #     hidden = layer.post_attention_layernorm(hidden)
    #     hidden = layer.mlp(hidden) + residual
    #     logits = modelNorm(hidden) @ Wlm.T
    #     return F.cross_entropy(logits.float(), gt_y).item()
    # return head_ablate_loss(head)
    
    for step in range(training_steps):
        alpha = F.softmax(a_param + mask, dim=-1)    
        
        logits = mlp_forward(alpha, head_idx, V_head, pos_list, original, residual_attn_in, Wo, Wlm)
        loss = F.cross_entropy(logits.float(), gt_y, reduction='mean')
        
        if step % 10 == 0:
            print("step", step, "loss:", loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return F.softmax(a_param + mask, dim=-1).detach().cpu()

head_idx = 25
pos_list = list([4091])
a_star = optimize_alpha_star(head_idx=head_idx, pos_list=pos_list, training_steps=200, lr=0.5, device=device)
a_star_2 = optimize_alpha_star(head_idx=head_idx, pos_list=pos_list, training_steps=200, lr=0.5, device=device)

# sparisity and entrophy
a_star = a_star.cpu()
entrophy = -(a_star * a_star.clamp_min(1e-8).log()).sum(dim=-1)
print("entrophy", entrophy.mean().item())

topk = 3
topk_mass = a_star.topk(topk, dim=-1).values.sum(dim=-1)
print("top{}_mass".format(topk), topk_mass.mean().item())

# KL divergence with original alpha
vanilla_alpha = attn['weights'][0][25, pos_list].cpu().float()
def KL_divergence(p, q):
    eps = 1e-12
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
    return kl_pq, kl_qp
kl_pq, kl_qp = KL_divergence(a_star, vanilla_alpha)
print("KL(a_star || vanilla_alpha)", kl_pq.mean().item())
print("KL(vanilla_alpha || a_star)", kl_qp.mean().item())

kl_pq, kl_qp = KL_divergence(a_star, a_star_2)
print("KL(a_star || a_star_2)", kl_pq.mean().item())
print("KL(a_star_2 || a_star)", kl_qp.mean().item())


# avg topk overlap

def topk_overlap(p, q, topk=10):
    p_topk_indices = torch.topk(p, k=topk, dim=-1).indices  # [n_pos, topk]
    q_topk_indices = torch.topk(q, k=topk, dim=-1).indices  # [n_pos, topk]

    overlap_counts = []

    for i in range(p_topk_indices.shape[0]):
        overlap = set(p_topk_indices[i].tolist()) & set(q_topk_indices[i].tolist())
        overlap_counts.append(len(overlap))
    # print("overlap_counts", overlap_counts)
    return sum(overlap_counts)/len(overlap_counts)/topk

topk=3
print("Average top-{} overlap: {}".format(topk, topk_overlap(a_star, vanilla_alpha, topk=topk)))
print("Average top-{} overlap: {}".format(topk, topk_overlap(a_star, a_star_2, topk=topk)))



# cosine in V space
head = head_idx
V = rope_qkv[L-1]['v'].float()  # [B, nh, seq, hd]
V_head = V[0][head]
V_1 = a_star @ V_head
V_2 = a_star_2 @ V_head
V_3 = vanilla_alpha @ V_head
cos = F.cosine_similarity(V_1, V_2, dim=-1)          # [n_pos]
rel = (V_1 - V_2).norm(dim=-1) / (V_1.norm(dim=-1) + 1e-12)
print("mix cosine mean/med:", cos.mean().item(), cos.median().item())
print("mix relerr mean/med:", rel.mean().item(), rel.median().item())
cos_vanilla = F.cosine_similarity(V_1, V_3, dim=-1)
rel_vanilla = (V_1 - V_3).norm(dim=-1) / (V_1.norm(dim=-1) + 1e-12)
print("vanilla cosine mean/med:", cos_vanilla.mean().item(), cos_vanilla.median().item())
print("vanilla relerr mean/med:", rel_vanilla.mean().item(), rel_vanilla.median().item())


# logits
vanilla = outputs["logits"]
vanilla_logits = vanilla[0][pos_list]
loss_baseline = F.cross_entropy(vanilla_logits.float(), gt_label[0, pos_list].to(vanilla_logits.device), reduction='none')
print("baseline loss", loss_baseline.mean().item())

device='cuda'
a_star = a_star.to(device)
residual_attn_in = last_layer_input['hidden_states'][0,pos_list].to(device) # [n_pos, hidden_size]
original = attn['output'][0,pos_list].permute(1,0,2).to(device) # [nh, n_pos, hd]
V_head = rope_qkv[L-1]['v'].to(device)[0][head_idx]  # [B, nh, seq, hd]
Wo = last_layer_param['self_attn.o_proj.weight'].to(device)
Wlm = kv["Wlm"].to(device)

logits = mlp_forward(a_star, head_idx, V_head, pos_list, original, residual_attn_in, Wo, Wlm)
loss_mix = F.cross_entropy(logits.float(), gt_label[0, pos_list].to(logits.device), reduction='none')
print("mix loss", loss_mix)

logits_2 = mlp_forward(a_star_2.to(device), head_idx, V_head, pos_list, original, residual_attn_in, Wo, Wlm)
loss_mix_2 = F.cross_entropy(logits_2.float(), gt_label[0, pos_list].to(logits_2.device), reduction='none')
print("mix_2 loss", loss_mix_2)

print("delta",(loss_mix_2 - loss_mix).abs())