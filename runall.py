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
for combo in combinations:
    args = dict(zip(keys, combo))
    cmd = [
        "CUDA_VISIBLE_DEVICES=1",  # 你的设备环境变量
        "python", "onlinetrain.py",
        f"--config-name=setattn_{task}",
        f"out_dir=out-{task}/k{int(args['k_mapping'])}_v{int(args['v_mapping'])}_s{int(args['smaller_sets'])}",
        f"wandb.log=true",
        f"wandb.run_name={task}_k{int(args['k_mapping'])}_v{int(args['v_mapping'])}_s{int(args['smaller_sets'])}",
        f"attn.k_mapping={str(args['k_mapping'])}",
        f"attn.v_mapping={str(args['v_mapping'])}",
        f"attn.smaller_sets={str(args['smaller_sets'])}",
    ]

    # 拼成单条命令字符串
    cmd_str = " ".join(cmd)
    print(f"\n=== Running: {cmd_str} ===\n")

    # 执行命令
    subprocess.run(cmd_str, shell=True, check=True)
