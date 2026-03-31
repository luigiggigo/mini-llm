from dataclasses import dataclass


@dataclass
class Config:
    data_path: str = "data/input.txt"
    out_dir: str = "out"

    batch_size: int = 32
    block_size: int = 256

    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

    learning_rate: float = 3e-4
    max_iters: int = 10000
    eval_interval: int = 200
    eval_iters: int = 50
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    device: str = "cuda"
    dtype: str = "float16"

    seed: int = 1337