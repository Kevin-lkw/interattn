import torch
from copy import deepcopy
from setattn.custom_linear_attn import SetAttention_Linear_fla, SetAttention_Linear_fla_fast
# 假设这两个类已经定义在当前作用域中
# from your_module import SetAttention_Linear_fla, SetAttention_Linear_fla_fast

# === 构造一个简易的 config ===
class DummyConfig:
    n_embd = 64
    n_head = 4
    dropout = 0.0
    bias = False
    level = 2
    levelrand = False
    k_mapping = True
    v_mapping = False
    smaller_sets = False
    feature_map = "elu"

# 初始化
config = DummyConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 两个模型
model_ref = SetAttention_Linear_fla(config).to(device)
model_fast = SetAttention_Linear_fla_fast(config).to(device)

# 同步参数（保证完全一致）
model_fast.load_state_dict(deepcopy(model_ref.state_dict()))

# === 构造输入 ===
B, T, C = 4, 128, config.n_embd
torch.manual_seed(42)
x = torch.randn(B, T, C, device=device)

import time
# === 前向计算 ===
with torch.no_grad():
    start = time.time()
    for _ in range(10):
        out_ref = model_ref(x)
    end = time.time()
    print(f"ref time: {end - start:.6f}s")
    start = time.time()
    for _ in range(10):
        out_fast = model_fast(x)
    end = time.time()
    print(f"fast time: {end - start:.6f}s")

# === 检查形状 ===
print(f"ref shape:  {out_ref.shape}")
print(f"fast shape: {out_fast.shape}")

# === 数值误差分析 ===
abs_diff = (out_ref - out_fast).abs()
max_err = abs_diff.max().item()
mean_err = abs_diff.mean().item()

print(f"\n✅ 最大误差: {max_err:.6e}")
print(f"✅ 平均误差: {mean_err:.6e}")

# === 判断是否等价 ===
if torch.allclose(out_ref, out_fast, atol=1e-5, rtol=1e-3):
    print("🎯 两个模型输出一致！")
else:
    print("⚠️ 输出存在差异，请检查中间计算。")

# === 选填：打印几个示例对比 ===
print("\n前5个元素对比：")
print("ref:", out_ref.view(-1)[:5])
print("fast:", out_fast.view(-1)[:5])