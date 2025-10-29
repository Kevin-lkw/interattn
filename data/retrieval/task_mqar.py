import torch as t
import numpy as np
# from torch import Tensor
# from data import task
"""
Adapted from https://github.com/HazyResearch/zoology/blob/main/zoology/data/associative_recall.py
"""
class mqar():
    """Task: multi-query associative recall, adapted to Copy interface.
       Input:  [K1 V1 K2 V2 ... Kn Vn | 0 ... 0 (with some query keys inserted)]
       Target: [-1 ... -1              | -1 ... -1 (only query positions = corresponding values)]
    """
    def __init__(self, device: str,
                 num_kv_pairs: int, vocab_size: int = 1024, power_a: float = 0.01,
                 random_non_queries: bool = True):
        self.num_kv_pairs = num_kv_pairs
        self.vocab_size = vocab_size
        self.power_a = power_a
        self.random_non_queries = random_non_queries
        self.device = device
        self.train_generator = np.random.default_rng(0)
        self.test_generator = np.random.default_rng(1)
        self.train_torch_generator = t.Generator(device=device)
        self.train_torch_generator.manual_seed(0)
        self.test_torch_generator = t.Generator(device=device)
        self.test_torch_generator.manual_seed(1)
    
    def sample_batch(self, split, batch_size, length: int, randomize: bool):
        gen = self.train_generator if split == 'train' else self.test_generator
        torch_gen = self.train_torch_generator if split == 'train' else self.test_torch_generator
        batch_size = batch_size or self.batch_size
        length = length or self.length

        assert length % 2 == 0, "input_seq_len must be even"
        assert self.vocab_size > length
        assert self.num_kv_pairs * 4 <= length


        # ----- Step 1: generate key-value pairs -----
        context_size = self.num_kv_pairs * 2
        key_vocab_size = self.vocab_size // 2
        key_choices = np.arange(1, key_vocab_size)
        value_choices = np.arange(key_vocab_size, self.vocab_size)

        def choice_arr(arr):
            return gen.choice(arr, size=self.num_kv_pairs, replace=False)
        keys_unshuffled = np.tile(key_choices, (batch_size, 1))
        keys = np.apply_along_axis(choice_arr, 1, keys_unshuffled)
        values_unshuffled = np.tile(value_choices, (batch_size, 1))
        values = np.apply_along_axis(choice_arr, 1, values_unshuffled)

        kvs = np.zeros((batch_size, context_size), dtype=np.int64)
        kvs[:, 0::2] = keys
        kvs[:, 1::2] = values

        # ----- Step 2: power-law distributed gaps for queries -----
        space = (length - context_size) // 2
        p = self.power_a * np.arange(1, space + 1) ** (self.power_a - 1)
        p = p / p.sum()
        x = np.stack([np.arange(space, dtype=int)] * batch_size)
        def choice_arr_p(arr):
            return gen.choice(arr, size=self.num_kv_pairs, replace=False, p=p)
        gaps = np.apply_along_axis(choice_arr_p, 1, x)

        # ----- Step 3: fill queries and labels -----
        queries = np.zeros((batch_size, length - context_size + 1), dtype=np.int64)
        np.put_along_axis(queries, (gaps * 2), values=keys, axis=1)
        examples = np.concatenate([kvs, queries], axis=1)
        labels = np.full((batch_size, length + 1), -1, dtype=np.int64)
        np.put_along_axis(labels, (gaps * 2) + context_size + 1, values=values, axis=1)

        X = t.tensor(examples[:, :-1], device=self.device)
        Y = t.tensor(labels[:, 1:], device=self.device)
        
        # replace all the 0 with random values if needed
        if self.random_non_queries:
            mask = (X == 0)
            X[mask] = t.randint(0, self.vocab_size, X.shape, generator = torch_gen, device=self.device)[mask]
        # import ipdb; ipdb.set_trace()
        return X, Y

    @property
    def input_size(self) -> int:
        return self.vocab_size

    @property
    def output_size(self) -> int:
        return self.vocab_size

    @property
    def vocabulary_size(self) -> int:
        return self.vocab_size

if __name__ == "__main__":
    print("=== 测试1: 单个类设置种子 ===")
    dataset = mqar(batch_size=1, length=20, randomize=False, device='cpu', num_kv_pairs=4)
    a1, b1 = dataset.sample_batch('test')
    print(f"MQAR 第一次: X={a1}, Y={b1}")

    dataset2 = mqar(batch_size=1, length=20, randomize=False, device='cpu', num_kv_pairs=4)
    print("\n=== 测试2: 重新生成 ===")
    a2, b2 = dataset2.sample_batch('test')
    print(f"MQAR 第二次: X={a2}, Y={b2}")

    print("\n=== 验证确定性 ===")
    print(f"MQAR 两次相同: {t.allclose(a1, a2)} and {t.allclose(b1, b2)}")