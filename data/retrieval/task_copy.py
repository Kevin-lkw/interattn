import torch as t
import torch.nn.functional as F
from data import task

class Copy(task.GeneralizationTask):
  """Task: <BOS> x1 ... xL <SEP> x1 ... xL-1 xL 
     Target: -1  -1 ... -1   x1  x2 ...  xL <EOS>
     POS:    0   1 ...  L    L+1 L+2 ... 2L 2L+1
     follows :Repeat After Me: Transformers are Better than State Space Models at Copying
  """
  def __init__(self, device: str):
    self.device = device
    self.range = 26 
    self.train_torch_generator = t.Generator(device=device)
    self.train_torch_generator.manual_seed(0)
    self.test_torch_generator = t.Generator(device=device)
    self.test_torch_generator.manual_seed(1)

  def sample_batch(self, split, batch_size: int, length: int, randomize: bool) -> task.Batch:
    """Creates sequences of the form [x1, ..., xL, 2, x1, ..., xL].
       Target Y masks the first (L+1) tokens as -1 and expects model
       to predict the copy part after the separator.
    """
    gen = self.train_torch_generator if split == 'train' else self.test_torch_generator
    if randomize :
      length = t.randint(1, length + 1, (), generator=gen,device=self.device).item()

    # Generate 01 sequences
    seq = t.randint(0, self.range, (batch_size, length), generator=gen, device=self.device)
    # alphabet: 0 ... rg-1
    # SEP token: rg
    # BOS token: rg+1
    # EOS token: rg+2
    separate = t.full((batch_size, 1), self.range, dtype=t.long, device=self.device)
    bos = t.full((batch_size, 1), self.range+1, dtype=t.long, device=self.device)
    X = t.cat([bos, seq, separate, seq], dim=1)  # shape: (B, 2L+2)
    # Target: model should output the copied sequence after SEP
    Y = -t.ones_like(X)
    Y[:, length+1:] = t.cat([seq, t.full((batch_size, 1), self.range+2, dtype=t.long, device=self.device)], dim=1)
    return X, Y

  @property
  def input_size(self) -> int:
    return self.range+3

  @property
  def output_size(self) -> int:
    return self.range+3  

  @property
  def vocabulary_size(self) -> int:
    return self.range+3  