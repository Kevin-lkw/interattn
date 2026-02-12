import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama import modeling_llama
from datasets import load_dataset

# load dataset
dataset_name="wikitext"
start_index= 0
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = '\n'.join([t for t in dataset['text'] if t.strip()])

# dataset_name="pg19"
# start_index= 0
# dataset = load_dataset("emozilla/pg19-test", split="test")
# texts = '\n'.join([t for t in dataset['text'] if t.strip()])

# import ipdb; ipdb.set_trace()
llama_model = "llama-7b-hf"
# load model
local_dir = f"/nfs-shared-2/models/llama/{llama_model}" 

tokenizer = AutoTokenizer.from_pretrained(
    local_dir,
    local_files_only=True,
    use_fast=False,          
)

model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    local_files_only=True,
    torch_dtype="auto",  
    device_map="auto",
    attn_implementation="eager", 
)
# model is LlamaForCausalLM

model.eval()

# tokenize input
prompt = texts
inputs = tokenizer(
    prompt,
    max_length = start_index + 4096,
    truncation=True,
    return_tensors="pt"
).to(model.device)

inputs = {k: v[:, start_index:start_index+4096] for k, v in inputs.items()}
kv_info = {}

def get_kv_hook(name):
    def hook(module, input, output):
        kv_info[name] = output.detach().cpu()
    return hook

# 遍历模型的每一层，找到 k_proj 和 v_proj
for name, module in model.named_modules():
    if "k_proj" in name or "v_proj" in name:
        module.register_forward_hook(get_kv_hook(name))

rope_qkv = {}

_orig_eager = modeling_llama.eager_attention_forward

def eager_wrapper(
    module,  # 注意：这里的 module 是 LlamaAttention 实例（源码里叫 self）
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    **kwargs,
):
    layer = getattr(module, "layer_idx", None)
    rope_qkv[layer] = {
        "q": query.detach().cpu(),
        "k": key.detach().cpu(),
        "v": value.detach().cpu(),
    }
    return _orig_eager(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=scaling,
        dropout=dropout,
        **kwargs,
    )

modeling_llama.eager_attention_forward = eager_wrapper


# 重新运行推理
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
Wo = model.model.layers[-1].self_attn.o_proj.weight.detach().cpu()
Wlm = model.lm_head.weight.detach().cpu()
    
save = {
    "model_dir": local_dir, 
    "model_config": model.config,
    "input": inputs,
    "before_rope": kv_info,
    "after_rope": rope_qkv,
    "Wo": Wo,
    "Wlm": Wlm,
}
torch.save(save, f"{llama_model}_{dataset_name}_st{start_index}.pt")
print(f"Saved kv and rope info to {llama_model}_{dataset_name}_st{start_index}.pt")