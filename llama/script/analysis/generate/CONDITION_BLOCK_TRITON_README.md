# Condition-Block Triton 优化

## 目标

优化 decode 阶段的 condition-block attention：

- 未展开 block：只计算 `q` 与 block representative 的 attention。
- 展开 block：只读取该 block 的 token K/V，并计算精确 attention。
- 不再像原 PyTorch 实现一样先计算所有 prompt token 的 `q @ k` 再 mask。

实现文件：`methods/condition_block_triton.py`。

## 核心优化

1. **真正的稀疏 K/V 访问**
   - 未选中的 block 不读取 token-level K/V。
   - selected block 以 16-token page 为单位加载到 SRAM。
   - representative、selected page 和 generated suffix 在同一个 online softmax 中合并。

2. **Tensor-Core stage2**
   - `block_size=16` 对应 16-token MMA tile。
   - selected page 的 QK 和 PV 使用 Triton `tl.dot`。
   - attention 采用 partition + reduce，思路类似 paged attention。

3. **Triton selection**
   - selection 拆成 partition stat s、reduce、finalize 三个 kernel。
   - 避免 PyTorch 生成 `upper/lower/delta/condition` 等大型中间张量。
   - selection mask 与原 PyTorch 实现完全一致。

4. **减少无效数据与同步**
   - prompt token pages 保持 BF16，不再缓存完整 FP32 K/V 副本。
   - 可通过 `CONDITION_BLOCK_SKIP_STATS=1` 关闭逐层 budget 统计及 `.item()` 同步。
   - 新增 `condition_block_triton` CLI method 和每条样本的生成耗时记录。

5. **融合 routing finalize 与 attention**
   - condition 计算完成后立即选择 representative 或 selected page。
   - 不再写回并重新读取 `selected` 和 `z_logits`。
   - 核心路径由 5 个 Triton kernel 降为 4 个。
   - `n_blocks/n_chunks/suffix_len` 禁止自动 specialization，避免不同输入长度反复编译。

## Sanity Check

```text
selection mask:                 完全一致
z_logits 最大绝对误差:         2.4e-7
attention 输出最大绝对误差:    约 2.1e-3
stats 开关前后生成结果:         5/5 完全一致
```

Triton 使用 BF16 Tensor Core，归约顺序也与 PyTorch 不同，因此不保证 token-level bitwise 一致。

## 速度结果

测试环境：Llama-3.1-8B-Instruct、BF16、单 batch、RTX PRO 6000 Blackwell、GovReport 前 5 条、`block_size=16`、`eps=0.1`、`max_new_tokens=128`。

| 实现 | 平均耗时 | 相对原 condition-block |
|---|---:|---:|
| Full attention | 2.87 s/sample | 2.47× |
| 原 `condition_block.py` | 7.08 s/sample | 1.00× |
| Triton，保留 stats | 4.39 s/sample | 1.61× |
| Triton，关闭 stats | 3.66 s/sample | 1.93× |

结论：Triton 相对原 condition-block 降低约 48% 耗时，但在 GovReport 的 4K–16K 上仍比 full attention 慢约 28%。以上 Triton 数字为 kernel cache 已预热的稳态结果；首次运行有一次性 JIT 编译成本。

GovReport 前 5 条 Rouge-L：

```text
原 condition-block: 22.12
Triton:              21.92
差值:                -0.20
完全相同预测:        4/5
```

局部 kernel 结果：

```text
10.7K、5% selected：stage2 Triton 约 0.034 ms，full SDPA 约 0.063 ms
10.7K selection：    Triton 约 0.034 ms，PyTorch eager 约 0.194 ms
64K hybrid attention：Triton 约 0.111 ms，full SDPA 约 0.194 ms
```

说明算法级稀疏确实生效，但局部 kernel 加速尚未完全转化为模型端到端加速。

## Profile：时间花在哪里

测试：GovReport 第一条，输入 10,733 tokens，生成 32 tokens，32 层；kernel 已预热，关闭 stats。

| 区域 | 调用次数 | CUDA 总时间 | 平均每次 | CPU 总时间 |
|---|---:|---:|---:|---:|
| Build clusters/summaries | 32 | 10.2 ms | 0.320 ms | 10.9 ms |
| Condition stats + reduce | 992 | 34.2 ms | 0.034 ms | 96.5 ms |
| Fused condition + attention 总区域 | 992 | 61.8 ms | 0.062 ms | 200.0 ms |

结论：

- **分 cluster 不是主要瓶颈**：每层只在首次 decode 构建一次，总计约 10 ms。
- 融合前 condition + sparse attention 的 CUDA 区域约为 154.8 ms；融合后完整区域约为 61.8 ms。
- 关闭 stats 的 GovReport 稳态时间由 3.86 降至 3.66 s/sample，端到端提升约 5%。
- 局部 attention 已不再是最大的 CUDA 成本，模型 MLP/GEMM 约占 CUDA 时间的 64%。

整个 profile 的 wall time由约 1.55 降至 1.46 s；粗略 GPU busy 比例由约 71% 提高到 76%。CUDA launch 数从约 48,918 降到 46,934，但仍出现约 109 ms 的 `Command Buffer Full` 等待。因此剩余问题仍主要是模型级细碎 kernel 和 Python/eager 调度，而不是 cluster summary 计算量。

## 当前瓶颈

1. **Kernel launch 太多**
   - 融合后每层仍需要 selection stats、selection reduce、fused attention partition、attention reduce。
   - 单 batch、`q_len=1` 时计算规模小，launch 和调度成本占比很高。

2. **Transformers eager hook 开销**
   - 每个 token、每层都经过 Python attention hook、shape 处理和临时 workspace 分配。
   - full SDPA 是高度融合的单 kernel 路径，更容易保持 GPU 连续执行。

3. **短中上下文尚未越过 crossover**
   - 4K–16K 时 full SDPA 已非常快，稀疏路径的固定开销大于节省的 K/V 带宽。
   - 约 64K 时单层 hybrid attention 才明显快于 full SDPA。

4. **Generated suffix 仍然精确计算**
   - suffix 随生成长度线性增长。
   - GovReport 等长生成任务后期会逐渐削弱 prompt 压缩收益。

## 后续优化方向

按优先级排序：

1. **Compiled custom op + Static KV cache + CUDA Graph**
   - 将完整 sparse attention 注册为一个 custom op。
   - 使用固定 KV 地址和预分配 workspace，减少 Python、allocator 和多 kernel replay 开销。
   - 这是端到端超过 full attention 最关键的方向。

2. **复用 workspace**
   - 按 layer/sample 预分配 selection 和 attention partial buffers。
   - 避免每层、每 token 创建临时 tensor。

3. **压缩较旧的 generated suffix**
   - 保留最近窗口 token 精确 attention。
   - 更早的生成 token 转为 block representative/selected-page 表示。

4. **按上下文长度动态路由**
   - 短上下文直接使用 full SDPA。
   - 长上下文、低 selected ratio 时启用 Triton condition-block。

## 使用方式

测速时关闭 stats：

```bash
CONDITION_BLOCK_SKIP_STATS=1 \
CONDITION_BLOCK_TRITON_CHUNK_BLOCKS=16 \
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=llama \
conda run -n nanogpt python -m script.analysis.generate.longbench.run \
  --device cuda:0 \
  --dataset gov_report \
  --method condition_block_triton \
  --condition-block-size 16 \
  --condition-eps 0.1 \
  --max-new-tokens 128
```

需要分析实际等效 budget 时不要设置 `CONDITION_BLOCK_SKIP_STATS=1`。
