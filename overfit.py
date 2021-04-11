from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

from data.dataset import CharDataset, build_synthetic_dataset
from transformer.pytorch.model import make_model, sample, make_gpt


batch_size = 4
num_samples = 4
num_epochs = 100
seq_len = 3
dmodel = 256
overfit = True
use_cuda = True
OFFSET = 1

torch.manual_seed(1233245245)

def build_dataloader(dataset: Dataset):
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(dataset,
        batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=True)
    return data_loader

def build_optimizer(model, base_lr):
    return torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.1)

def run_epoch(data_loader, model, loss_func):
    losses = []
    for batch_idx, data_batch in enumerate(data_loader):
        x, y = data_batch.x, data_batch.y
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        out = model(x, y, src_mask=None, tgt_mask=None)
        logits = model.generator(out)
        losses.append(loss_func(logits.view(-1, logits.shape[-1]), y.view(-1)))
    total_loss = torch.stack(losses).sum()
    return total_loss

def run_gpt_epoch(data_loader, model, loss_func):
    losses = []
    for batch_idx, data_batch in enumerate(data_loader):
        x, y = data_batch.x, data_batch.y
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        logits = model(x)
        losses.append(loss_func(logits.view(-1, logits.shape[-1]), y.view(-1)))
    total_loss = torch.stack(losses).sum()
    return total_loss

def greedy_decode(model):
    pass

def build_dataset(block_size: int):
    limit_len = batch_size * num_samples if overfit else None
    return CharDataset(block_size, limit_len)

def num_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(
    model: torch.nn.Module,
    train_data_loader: DataLoader):
    # train_dataset = build_synthetic_dataset(block_size, num_samples)
    loss_func = F.cross_entropy
    optim = build_optimizer(model, base_lr=3e-4)

    for epoch in range(num_epochs):
        model.train()
        loss = run_gpt_epoch(train_data_loader, model, loss_func)
        model.zero_grad()
        loss.backward()
        optim.step()
        _eval(model, train_data_loader)
        print(f"Loss epoch {epoch}: {loss}")

def _eval(model: torch.nn.Module, data_loader: DataLoader, steps: Optional[int] = 1):
    dataset = data_loader.dataset
    for batch_idx, data_batch in enumerate(data_loader):
        x, y = data_batch.x, data_batch.y
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        pred = sample(model, x, steps, seq_len)
        _pred = dataset.decode(pred)
        _pred = [i[OFFSET:] for i in _pred]
        print(f"  x: {dataset.decode(x)}, gt: {dataset.decode(y)} pred: {_pred}")

if __name__ == "__main__":
    train_dataset = build_dataset(block_size=seq_len)
    train_data_loader = build_dataloader(train_dataset)
    model_encdec = make_model(train_dataset.vocab_size, train_dataset.vocab_size, d_model=dmodel)
    model_gpt = make_gpt(train_dataset.vocab_size, d_model=dmodel)
    print(f'Model num params (EncDec): {num_parameters(model_encdec)}')
    print(f'Model num params (GPT): {num_parameters(model_gpt)}')
    if use_cuda:
        model_gpt.cuda()
    train(model_gpt, train_data_loader)
    # _eval(model_gpt, train_data_loader)