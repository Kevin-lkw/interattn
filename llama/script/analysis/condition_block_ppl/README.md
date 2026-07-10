# Condition-block PPL

这个包包含 condition-block 的分析、perplexity（PPL）评测以及相关 baseline。
这里的 runner 通过 teacher-forced forward 计算 NLL/PPL，不执行自回归 generation。
自回归生成、LongBench、RULER 和 decode kernel 位于相邻的
`condition_block_gen/`。

## 主方法

| 文件 | 作用 |
| --- | --- |
| `condition_block.py` | 单层分析入口：计算各 cluster 的 condition，并按阈值决定压缩或展开 |
| `runner_cond_block.py` | condition-block 多层 PPL runner |
| `condition_block_hierarchy.py` / `runner_cond_block_hierarchy.py` | 层次化 block 变体 |
| `condition_block_single.py` | 单次 forward 实现与 sanity check |
| `condition_block_optim.py` | 更高效的 streaming/SDPA 实现 |
| `condition_block_corr.py` | condition 与真实误差的相关性分析 |

`condition.py` 是早期基于 H2O cluster 的 condition 分析，保留用于追溯
condition 的来源和诊断逻辑。

## Cluster 变体与 baseline

- `condition_ksim_cluster.py` / `runner_ksim_cluster.py`：仅使用 K-similarity cluster。
- `condition_ksim_condition.py` / `runner_condition_ksim_cluster.py`：在 K-similarity cluster 上应用 condition threshold。
- `runner_quest.py`：QUEST PPL baseline。
- `multisample/`：WikiText 多样本 PPL 对比，包含 condition-block、attention top-k、H2O、StreamLLM 和 QUEST。

## Bound 研究

- `condition_bound/`：原始 condition bound、Bennett 变体、PPL sweep 和 kernel microbenchmark。
- `delta_bound/`：更紧 delta 估计的候选方案与 term decomposition。

这些目录现在都是正常 Python package，建议使用 `-m` 运行，不再依赖从具体脚本路径启动。

## 运行方式

从 `llama` 目录运行：

```bash
# 单层 condition 分析
python -m script.analysis.condition_block_ppl.condition_block --help

# 多层 PPL sweep
python -m script.analysis.condition_block_ppl.runner_cond_block --help

# 多样本评测
python -m script.analysis.condition_block_ppl.multisample.run_condition_block --help

# condition-bound 实验
python -m script.analysis.condition_block_ppl.condition_bound.hybrid_guarantee --help
```

默认结果路径和结果文件名没有随源码目录迁移而改变，因此已有结果仍可直接读取。

## 共享基础设施

PPL 与 generation 两侧共同复用父目录的 `attention.py`、`online_routing.py`、
`runtime.py`、`runner_utils.py`、`experiment_utils.py`、`sanity.py` 和
`context.py`。这些文件不属于某一个 phase，因此保留在 `analysis/` 根目录。
