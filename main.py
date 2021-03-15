import torch
import torch.nn.functional as F

from exploration.dataset_exploration import build_dataset, build_synthetic_dataset
from transformer.pytorch.model import make_model

batch_size = 10
num_epochs = 50
seq_len = 8
dmodel = 64

def build_optimizer(model, base_lr):
    return torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9)

def run_epoch(dataset, model, loss_func):
    x, y = dataset.batch(batch_size)
    out = model(x, y, src_mask=None, tgt_mask=None)
    logits = model.generator(out)
    loss = loss_func(logits.view(-1, logits.shape[-1]), y.view(-1))
    return loss

def greedy_decode(model):
    pass

# train_dataset = build_synthetic_dataset(block_size, num_samples) 
train_dataset = build_dataset(block_size=seq_len)

model = make_model(train_dataset.vocab_size, train_dataset.vocab_size, d_model=dmodel)
loss_func = F.cross_entropy
optim = build_optimizer(model, base_lr=3e-4)

for epoch in range(num_epochs):
    model.train()
    loss = run_epoch(train_dataset, model, loss_func)
    model.zero_grad()
    loss.backward()
    optim.step()
    print(f"Loss epoch {epoch}: {loss}")

model.eval()
