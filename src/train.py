import os
from pathlib import Path
import torch

from config import Config
from data import ByteDataset
from model import GPT


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def estimate_loss(model, dataset, cfg):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = dataset.get_batch(split, cfg.batch_size, cfg.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA non disponibile, passo a CPU")
        cfg.device = "cpu"
        cfg.dtype = "float32"

    dataset = ByteDataset(cfg.data_path, cfg.block_size)

    model = GPT(
        vocab_size=dataset.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.device == "cuda" and cfg.dtype == "float16"))

    for step in range(cfg.max_iters):
        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, dataset, cfg)
            print(f"step {step}: train {losses['train']:.4f} | val {losses['val']:.4f}")

            ckpt = {
                "model": model.state_dict(),
                "config": cfg.__dict__,
            }
            torch.save(ckpt, Path(cfg.out_dir) / "ckpt.pt")

        x, y = dataset.get_batch("train", cfg.batch_size, cfg.device)

        optimizer.zero_grad(set_to_none=True)

        if cfg.device == "cuda" and cfg.dtype == "float16":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

    torch.save({"model": model.state_dict(), "config": cfg.__dict__}, Path(cfg.out_dir) / "final.pt")
    print("Training completato")


if __name__ == "__main__":
    main()
