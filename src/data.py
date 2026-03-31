from pathlib import Path
import torch


class ByteDataset:
    def __init__(self, path: str, block_size: int):
        self.path = Path(path)
        self.block_size = block_size

        text = self.path.read_text(encoding="utf-8")
        data = text.encode("utf-8", errors="ignore")
        self.data = torch.tensor(list(data), dtype=torch.long)

        self.vocab_size = 256

        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split: str, batch_size: int, device: str):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return bytes(tokens).decode("utf-8", errors="ignore")
