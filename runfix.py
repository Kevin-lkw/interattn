import itertools
import subprocess

# 三个布尔参数
params = {
    "k_mapping": [False, True],
    "v_mapping": [False, True],
    "smaller_sets": [False, True],
}

# 所有组合（8个）
keys = list(params.keys())
combinations = list(itertools.product(*params.values()))
task = "copy"

for level in range(1):
    name = f"fix-L{level}-k0v0s0"
    cmd = [
        "CUDA_VISIBLE_DEVICES=7",  # 你的设备环境变量
        "python", "onlinetrain.py",
        f"--config-name=setattn_{task}",
        f"out_dir=out-{task}/vanilla",
        f"wandb.log=true",
        f"wandb.run_name={task}_vanilla",
        "model.n_layer=8","model.n_head=8","model.n_embd=128",
        "attn.type=vanilla",
        "attn.levelrand=false",
        f"attn.level={level}",
    ]

    # 拼成单条命令字符串
    cmd_str = " ".join(cmd)
    print(f"\n=== Running: {cmd_str} ===\n")

    # 执行命令
    subprocess.run(cmd_str, shell=True, check=True)
