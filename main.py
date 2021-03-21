import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

from exploration.dataset_exploration import CharDataset, build_synthetic_dataset
from transformer.pytorch.model import make_model


batch_size = 10
num_epochs = 50
seq_len = 8
dmodel = 64
overfit = True

def build_dataloader(dataset: Dataset, sampler: Sampler):
    data_loader = DataLoader(dataset,
        batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=True)
    return data_loader

def build_optimizer(model, base_lr):
    return torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9)

def run_epoch(data_loader, model, loss_func):
    import pdb; pdb.set_trace()
    loss = torch.zeros((1))
    for batch_idx, data_batch in enumerate(data_loader):
        x, y = data_batch.x, data_batch.y
        out = model(x, y, src_mask=None, tgt_mask=None)
        logits = model.generator(out)
        loss += loss_func(logits.view(-1, logits.shape[-1]), y.view(-1))
    return loss

def greedy_decode(model):
    pass

def build_dataset(block_size: int):
    # you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
     # don't worry we won't run out of file handles
    limit_len = batch_size if overfit else None
    return CharDataset(block_size, limit_len) # one line of poem is roughly 50 characters

def train():
    # train_dataset = build_synthetic_dataset(block_size, num_samples)
    train_dataset = build_dataset(block_size=seq_len)
    train_data_loader = build_dataloader(train_dataset)

    model = make_model(train_dataset.vocab_size, train_dataset.vocab_size, d_model=dmodel)
    loss_func = F.cross_entropy
    optim = build_optimizer(model, base_lr=3e-4)

    # import pdb; pdb.set_trace()
    for epoch in range(num_epochs):
        model.train()
        loss = run_epoch(train_data_loader, model, loss_func)
        model.zero_grad()
        loss.backward()
        optim.step()
        print(f"Loss epoch {epoch}: {loss}")

    model.eval()

if __name__ == "__main__":
    train()