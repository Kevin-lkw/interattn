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

6. **持久 runner 与 workspace 复用**
   - 每个样本只创建一次 decode runner 和 monkeypatch context。
   - 每层复用 selection cache、partial softmax 和输出 workspace。
   - 避免每个 token 重建 module 映射和分配 Triton 临时 tensor。

7. **消除 decode dtype 与 suffix 拷贝 kernel**
   - Triton 直接按真实 stride 读取非连续 suffix KV view，不再逐层调用 `.contiguous()`。
   - Q 保持 BF16，进入 Triton 后在寄存器中转 FP32；reduce 直接写 BF16 输出。
   - 关闭 stats 时不再逐层创建 `pos_tensor`。
   - 相比 workspace 版，GovReport 端到端再提升约 10.7%。

8. **32-token page 与代码拆分**
   - Fast path 支持 `block_size=16/32`；32-token selected page 使用两个 16-token MMA tile。
   - page size 与 Tensor Core tile 解耦，representative、两段 selected tokens 和 suffix 共享 online softmax。
   - 顶层文件变为兼容 wrapper；实现拆为 `condition_block_triton_impl/core.py`、`page_attention.py` 和 `legacy.py`。

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
| Triton，保留 stats | 4.24 s/sample | 1.67× |
| Triton，关闭 stats | 3.20 s/sample | 2.21× |

结论：Triton 相对原 condition-block 降低约 55% 耗时；在 GovReport 的 4K–16K 上与 full attention 的差距缩小到约 11%。以上 Triton 数字为 kernel cache 已预热的稳态结果；首次运行有一次性 JIT 编译成本。

### GPU utilization 与 block size

同一张 RTX PRO 6000、同一批 GovReport 5×128，使用 NVML 约 100 ms 间隔采样：

| 实现 | 平均利用率 | 中位数 | 利用率 ≥90% 的采样占比 |
|---|---:|---:|---:|
| Full attention | 89.7% | 92% | 92.7% |
| Condition-block Triton | 84.8% | 92% | 82.2% |

Full attention 已经大部分时间处于高利用率；Triton 路径仍有更多调度气泡，但并非 GPU 完全空闲。当前测试的 `block_size=16` 与 Triton MMA tile 一致。实测等效 budget 为约 9.9%–12.2%；由

```text
equiv_ratio = selected_ratio + (1 - selected_ratio) / 16
```

反推，约 4%–6% blocks 被展开。说明稀疏度已经较高，block 太小不是当前主要问题。更大的 block 可以减少 representative/selection 数量，但会让 range bound 更松、单次展开读取更多 token；它最多直接影响当前约 4.3% 的 condition+sparse CUDA 时间，应作为独立精度/速度实验，而不是首要端到端优化。B32 fast path 与实测结果见下节。

### Block size 32 实验

设置：GovReport 前 5 条、128 tokens、`eps=0.1`、stats-off 性能；budget 来自相同配置的 stats-on run。

| Block size | 时间 | 等效 budget | 估算展开 block 比例 | Rouge-L |
|---:|---:|---:|---:|---:|
| 16 | 3.207 s/sample | 11.28% | 5.4% | 21.92 |
| 32 | 3.223 s/sample | 14.38% | 11.6% | 24.52 |

B32 比 B16 慢约 0.5%，基本持平。虽然 representative 数量减半，但更宽 cluster 的 range bound 更松，展开 block 比例约翻倍，实际 token budget 反而增加。5 条的小样本 Rouge-L 不能作为精度结论，但未观察到明显退化。

Synthetic fused-vs-dense sanity：

```text
B16 max_abs = 1.73e-3, mean_abs = 2.63e-4
B32 max_abs = 1.42e-3, mean_abs = 2.52e-4
两者均无 NaN/Inf
```

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

测试：GovReport 第一条，输入 10,733 tokens，生成 32 tokens，32 层；kernel 已预热，关闭 stats。下表为当前 3.20 s/sample 稳定版重新 profile 的结果。

| 区域 | 调用次数 | CUDA self time | CUDA 占比 |
|---|---:|---:|---:|
| 模型 GEMM/GEMV (`aten::mm`) | 7,200 | 709.0 ms | 64.7% |
| Prefill FlashAttention | 32 | 97.7 ms | 8.9% |
| Command Buffer Full 等待 | 215 | 108.5 ms | 9.9% |
| `aten::cat` 总计 | 4,225 | 63.6 ms | 5.8% |
| 其中 DynamicCache K/V append | 1,984 | 47.5 ms | 4.3% |
| Condition stats + reduce | 992 × 2 | 23.5 ms | 2.1% |
| Sparse attention + reduce | 992 × 2 | 23.5 ms | 2.1% |

结论：

- **分 cluster 不是主要瓶颈**：每层只在首次 decode 构建一次，总计约 10 ms。
- 当前 condition 与 sparse attention 四个 Triton kernel 合计约 47 ms，仅占 CUDA 时间 4.3%；继续只优化 sparse kernel 的端到端上限已经很低。
- 关闭 stats 的 GovReport 稳态时间由 3.86 降至 3.66 s/sample，端到端提升约 5%。
- runner/workspace 复用进一步将 3.66 降至 3.58 s/sample，提升约 2.2%。
- 去掉 Q/output cast 与 suffix contiguous copy 后由 3.58 降至 3.20 s/sample，再提升约 10.7%；5/5 预测与优化前一致。
- 局部 attention 已不再是主要瓶颈，模型投影与 MLP GEMM/GEMV 占 CUDA 时间约 65%。

当前 profile 的 CPU self time 约 0.98 s、CUDA self time 约 1.10 s，仍出现约 109 ms 的 `Command Buffer Full`。因此剩余问题主要是模型 GEMV/MLP、逐层 elementwise kernel、KV cache append 和模型级 eager 调度，而不是 cluster summary 或 sparse attention 计算量。

## 当前瓶颈

1. **模型 GEMM/GEMV 是第一瓶颈**
   - Q/K/V/O projection 和 MLP 占 CUDA 时间约 65%。
   - batch=1、`q_len=1` 时大量操作退化为小 GEMV，GPU 很难完全吃满。

2. **模型级 eager/elementwise 调度**
   - RMSNorm、RoPE、residual、SiLU/multiply 等仍是大量独立 kernel。
   - `Command Buffer Full` 约占 10%，说明只 graph 四个 attention kernel 的粒度太小。

3. **DynamicCache 每 token 复制 K/V**
   - 1,984 次 K/V `cat` 正好对应 `31 decode steps × 32 layers × K/V`，CUDA 约 47.5 ms。
   - 这是明确但上限约 4–5% 的次级优化点。

4. **StaticCache 单独使用会回退**
   - 可通过 `CONDITION_BLOCK_STATIC_CACHE=1` 启用；8-token sanity 完全一致，长生成可能因数值顺序产生轨迹分叉。
   - 无 CUDA Graph 时当前实测由 3.20 回退到 3.60 s/sample，因此默认仍使用 DynamicCache。
   - StaticCache 的价值是固定指针，为后续 graph capture 服务，而不是单独加速。

5. **短中上下文尚未越过 crossover**
   - 4K–16K 时 full SDPA 已非常快，稀疏路径的固定开销大于节省的 K/V 带宽。
   - 约 64K 时单层 hybrid attention 才明显快于 full SDPA。

6. **Generated suffix 仍然精确计算**
   - suffix 随生成长度线性增长。
   - GovReport 等长生成任务后期会逐渐削弱 prompt 压缩收益。

## 后续优化方向

按优先级排序：

1. **先做模型级融合，而不是继续压 sparse kernel**
   - 将 condition-block attention 注册为 `torch.library.custom_op`，替代运行时 monkeypatch，使 Dynamo 能把它视为单个可编译算子。
   - 编译完整 decoder layer/step，融合 RMSNorm、RoPE、residual、SiLU/multiply 等 elementwise 路径。
   - 优先试 fused QKV projection 和 fused MLP gate/up projection；这直接针对约 65% 的 GEMM/GEMV 主耗时，并减少每层 launch 数。

### 已撤回实验：per-layer CUDA Graph

实验曾使用固定 Q staging buffer 和 device-side `suffix_len`，把每层的 selection
与 sparse attention 四个 Triton kernel 捕获为一个 graph。GovReport 5×128 sanity
与普通 StaticCache 5/5 一致，但速度为：

```text
DynamicCache 稳定版：       3.20 s/sample
StaticCache：               3.60 s/sample
StaticCache + layer graph： 3.68 s/sample
```

结论：只捕获 attention 子图粒度太小，graph replay、Q staging 和每样本 32 次
首次捕获抵消了 launch 收益。相关实现已撤回，仅保留结果。若以后继续 CUDA
Graph，必须捕获完整 decoder step，而不是 per-layer attention 子图。

2. **实现动态视图的预分配 KV cache**
   - 预分配底层 K/V storage，但每步只向模型暴露 `:visible_len` view。
   - 目标是消除 DynamicCache 的 K/V `cat`，同时避免 HF StaticCache 处理完整 capacity 的回退。
   - 预期收益上限约 4–5%，应作为独立小实验验证。

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
