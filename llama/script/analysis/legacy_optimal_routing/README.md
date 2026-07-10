# Legacy optimal-routing experiments

这个目录保存早期用于回答“optimal routing 为什么有效、能否被简单规则逼近”的探索代码。
它们是研究过程与诊断工具，不是当前 condition-block 方法的主实现。

当前主线位于父目录：

- `condition_block.py`：按 cluster 计算 condition，并根据阈值决定压缩或展开。
- `runner_cond_block.py`：condition-block 的多层 PPL runner。
- `condition_block_hierarchy.py` / `runner_cond_block_hierarchy.py`：层次化变体。
- `condition_block_single.py`、`condition_block_optim.py`、`condition_block_corr.py`：单次 forward、efficient 实现和相关性分析。
- `condition_ksim_*.py` / `runner_*ksim_cluster.py`：K-similarity cluster 变体。

新 condition-block 变体应继续放在父目录，不要放进本目录。

## 实验与 runner 对照

| 研究问题 | 单层分析 / compare | 多层 PPL runner |
| --- | --- | --- |
| 直接优化 routing 的上界 | `compare_optimal_routing.py` | `runner.py`；`runner_inter.py` 只替换最重要的 routing 差异 |
| 只优化 heavy-hitter routing | `compare_optimal_routing_hh.py` | `runner_inter_optimal_hh.py` |
| cluster count / mean-K / mean-V 近似 | `compare_count.py`、`compare_count_all.py`、`compare_count_avgK.py`、`compare_count_avgKV.py`、`compare_count_avgV.py`、`compare_count_vark.py` | `runner_count_all.py`、`runner_inter_avgKV.py` |
| cluster 内最优 V 标量及 oracle 上界 | `compare_count_optV.py`、`compare_count_optV_all.py`、`compare_V.py`、`compare_original.py` | `runner_inter_optV.py`、`runner_inter_original.py` |
| count-all 与 full attention 的误差分解 | `compare_error.py`、`compare_error_unnorm.py`、`compare_error_clusterplot.py`、`compare_error_qscan.py`、`compare_error_fixed_cluster_qscan.py` | `runner_inter_key_error.py`、`runner_inter_value_error.py`、`runner_inter_kv_error.py`；公共实现位于 `runner_inter_error_common.py` |
| 用简单 QK 参数化逼近 routing | `compare_q_bias.py`、`compare_q_linear.py`、`compare_q_temp.py` | `runner_inter_q_bias.py`、`runner_inter_q_linear.py`、`runner_inter_q_temp.py` |

以下脚本没有一一对应的 runner：

- `compare_count_oracle.py` / `compare_count_oracle_all.py`：oracle cluster 修正。
- `compare_db.py`：不同初始化得到的 optimal routing 对比。
- `compare_overlap.py`、`compare_routing.py`、`compare_sparsity.py`：可视化和统计 helper。
- `compare_count_summary_plot.py`：汇总多个 compare 结果。
- `check_avgkv_all_layers.py`：跨层检查 avgKV 是否接近 full-attention V。
- `generate_data_h2o.py`：生成带 oracle per-key bias 的 H2O 训练数据。

`plot.py` 是旧实验的统一绘图入口；`ada_plot.py` 和 `plot_inter*.py` 是更早的专用绘图脚本，保留用于复现实验。

## 运行方式

在 `llama/script` 目录下使用 module 方式运行。移动后的 module path 增加了
`legacy_optimal_routing`：

```bash
cd llama/script

# 单层 direct-optimal 对比
python -m analysis.legacy_optimal_routing.compare_optimal_routing --help

# 旧的 online optimal-routing 主 runner
python -m analysis.legacy_optimal_routing.runner --help

# avgKV 的多层 PPL 实验
python -m analysis.legacy_optimal_routing.runner_inter_avgKV --help

# 统一绘图入口
python -m analysis.legacy_optimal_routing.plot --help
```

默认结果路径仍相对于启动目录解析；这次整理只改变源码 module path，没有迁移历史结果，也没有修改结果文件名。

## 共享基础设施

本目录仍复用父目录中的通用组件：

- `attention.py`：mask、routing alpha 与 attention helper。
- `online_routing.py`：逐层 capture / patch 执行框架。
- `runtime.py`：模型、tokenizer、context 加载和 layer 选择。
- `runner_utils.py`：runner 参数、结果路径与指标 helper。
- `experiment_utils.py`：单层实验与 condition 分析共用的参数、prefix patch 和绘图 helper。
- `sanity.py` / `context.py` / `config.py`：公共数据结构与基础工具。

这些共享模块留在父目录，是因为 condition-block 主线与 legacy 实验都依赖它们。

## 历史命名说明

文件名与输出 tag 保留原样，便于读取已有结果。例如
`compare_count_avgV.py` 内部仍使用历史 tag `compare_count_sumV`。本次整理不顺手修正这些命名，以免破坏旧结果路径和分析脚本。
