import torch

from config import Config
from data import ByteDataset
from model import GPT


def main():
    cfg = Config()
    dataset = ByteDataset(cfg.data_path, cfg.block_size)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"

    ckpt = torch.load("out/ckpt.pt", map_location=cfg.device)

    model = GPT(
        vocab_size=dataset.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    ).to(cfg.device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = "Ciao"
    context = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device=cfg.device)

    out = model.generate(context, max_new_tokens=300, temperature=0.8, top_k=50)
    print(dataset.decode(out[0]))


if __name__ == "__main__":
    main()
