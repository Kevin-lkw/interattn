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
| Triton，保留 stats | 4.59 s/sample | 1.54× |
| Triton，关闭 stats | 3.86 s/sample | 1.84× |

结论：Triton 相对原 condition-block 降低约 46% 耗时，但在 GovReport 的 4K–16K 上仍比 full attention 慢约 35%。

GovReport 前 5 条 Rouge-L：

```text
原 condition-block: 22.12
Triton:              21.72
差值:                -0.40
完全相同预测:        3/5
```

局部 kernel 结果：

```text
10.7K、5% selected：stage2 Triton 约 0.034 ms，full SDPA 约 0.063 ms
10.7K selection：    Triton 约 0.034 ms，PyTorch eager 约 0.194 ms
64K hybrid attention：Triton 约 0.111 ms，full SDPA 约 0.194 ms
```

说明算法级稀疏确实生效，但局部 kernel 加速尚未完全转化为模型端到端加速。

## 当前瓶颈

1. **Kernel launch 太多**
   - 每层需要 selection stats、selection reduce、selection finalize、attention partition、attention reduce。
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

1. **融合 selection finalize 与 stage2 partition**
   - selection 得到 condition 后直接计算 representative/selected-page attention。
   - 避免写回和重新读取 `selected`、`z_logits`。

2. **Compiled custom op + Static KV cache + CUDA Graph**
   - 将完整 sparse attention 注册为一个 custom op。
   - 使用固定 KV 地址和预分配 workspace，减少 Python、allocator 和多 kernel replay 开销。
   - 这是端到端超过 full attention 最关键的方向。

3. **复用 workspace**
   - 按 layer/sample 预分配 selection 和 attention partial buffers。
   - 避免每层、每 token 创建临时 tensor。

4. **压缩较旧的 generated suffix**
   - 保留最近窗口 token 精确 attention。
   - 更早的生成 token 转为 block representative/selected-page 表示。

5. **按上下文长度动态路由**
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
