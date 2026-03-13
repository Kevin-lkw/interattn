import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama import modeling_llama, LlamaForCausalLM
from datasets import load_dataset

## load dataset
dataset_name="wikitext"
start_index= 0
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = '\n'.join([t for t in dataset['text'] if t.strip()])

# dataset_name="pg19"
# start_index= 0
# dataset = load_dataset("emozilla/pg19-test", split="test")
# texts = '\n'.join([t for t in dataset['text'] if t.strip()])

# dataset_name="gsm8k"
# start_index= 0
# dataset = load_dataset("openai/gsm8k", "main", split="test")
# sample = dataset[0]
# q = sample['question']
# a = sample['answer']
# full_text = f"Question: {q}\nAnswer: {a}"

# # 1. 先 tokenize 只有问题的部分，用来确定长度
# question_part = f"Question: {q}\nAnswer: "
# question_encoding = tokenizer(question_part, return_tensors="pt", add_special_tokens=True)
# answer_start_index = question_encoding["input_ids"].shape[1] # 这里的 index 就是 Answer 开始的位置

# # 2. tokenize 整个序列用于 Forward
# inputs_all = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(model.device)


llama_model = "meta-llama/Llama-2-7b-hf"
model_name = "llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(
    llama_model,
    use_fast=False,        
)
dtype = torch.float32
model = AutoModelForCausalLM.from_pretrained(
    llama_model,
    dtype=dtype,  
    device_map="auto",
    attn_implementation="eager",
)
model:LlamaForCausalLM

model.eval()

# tokenize input
prompt = texts
inputs_origin = tokenizer(
    prompt,
    max_length = start_index + 8192,
    truncation=True,
    return_tensors="pt"
).to(model.device)
context_len = 4096
inputs = {k: v[:, start_index:start_index+context_len] for k, v in inputs_origin.items()}
gt_label = inputs_origin["input_ids"][:, start_index+1:start_index+context_len+1]

kv_info = {}

def get_kv_hook(name):
    def hook(module, input, output):
        kv_info[name] = output.detach().cpu()
    return hook

# 遍历模型的每一层，找到 k_proj 和 v_proj
for name, module in model.named_modules():
    if "k_proj" in name or "v_proj" in name:
        module.register_forward_hook(get_kv_hook(name))

# hook input for each layer
layer_input = {}
def layer_input_hook(layer_idx):
    def hook(module, inp, out):
        layer_input[layer_idx] = inp[0].detach().cpu()
    return hook

# h = model.model.layers[-1].register_forward_hook(last_layer_input_hook)
for i, layer in enumerate(model.model.layers):
    layer.register_forward_hook(layer_input_hook(i))

rope_qkv = {}
attn = {}
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
    attn_output, attn_weights =  _orig_eager(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=scaling,
        dropout=dropout,
        **kwargs,
    )
    attn[layer] = {
        "output": attn_output.detach().cpu(),
    }
    return attn_output, attn_weights

modeling_llama.eager_attention_forward = eager_wrapper


# 重新运行推理
with torch.no_grad():
    outputs = model(**inputs,labels = inputs['input_ids'], use_cache=True)
    # import ipdb; ipdb.set_trace()
Wnorm = model.model.norm.weight.detach().cpu()
Wlm = model.lm_head.weight.detach().cpu()
last_layer = model.model.layers[-1]

save = {
    "model_dir": llama_model, 
    "model_config": model.config,
    "input": inputs,
    "output": outputs,
    "after_rope": rope_qkv,
    "attention_output": attn,
    "layer_input": layer_input,
    "gt_label": gt_label,
}
torch.save(save, f"../{model_name}_{dataset_name}_st{start_index}.pt")
print(f"Saved to ../{model_name}_{dataset_name}_st{start_index}.pt")