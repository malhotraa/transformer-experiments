import unittest
import torch
from data.dataset import CharDataset, DataSample, DataBatch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler


class TestBasicCharDataset(unittest.TestCase):

    def setUp(self):
        self.block_size = 5
        self.limit_len = 10
        self.dataset = CharDataset(self.block_size, self.limit_len)
    
        sample = DataSample(
            x=torch.randint(high=self.limit_len, size=(self.block_size,), dtype=torch.long),
            y=torch.randint(high=self.limit_len, size=(self.block_size,), dtype=torch.long)
        )
        self.samples = [sample, sample]
        self.n = len(self.samples)

    def test_limit_dataset(self):
        self.assertEqual(len(self.dataset), self.limit_len)
  
    def test_getitem(self):
        item = self.dataset[0]
        self.assertIsInstance(item, DataSample)
        x, y = item
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, (self.block_size,))
        self.assertEqual(y.shape, (self.block_size,))
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)
    
    def test_collate(self):
        batch = self.dataset.collate_fn(self.samples)
        self.assertIsInstance(batch, DataBatch)
        self.assertEqual(len(batch), self.n)
        x, y = batch
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, (self.n, self.block_size))
        self.assertEqual(y.shape, (self.n, self.block_size))
        
class TestDataLoader(unittest.TestCase):
    
    def setUp(self):
        torch.manual_seed(42)
        self.block_size = 5
        self.limit_len = 10
        self.batch_size = 2
        self.dataset = CharDataset(self.block_size, self.limit_len)
        self.sequential_sampler = SequentialSampler(self.dataset)
        self.random_sampler = RandomSampler(self.dataset)
    
    def test_data_loader_sequential_sampler(self):
        data_loader = DataLoader(
            self.dataset,
            self.batch_size,
            sampler=self.sequential_sampler,
            collate_fn=self.dataset.collate_fn,
            drop_last=True)

        self.assertEqual(len(list(data_loader)), len(self.dataset) // self.batch_size)
        zeroth_batch = self.dataset.collate_fn([self.dataset[0], self.dataset[1]])
        for batch_idx, data_batch in enumerate(data_loader):
            if batch_idx == 0:
                self.assertTrue(torch.allclose(data_batch.x, zeroth_batch.x))
                self.assertTrue(torch.allclose(data_batch.y, zeroth_batch.y))
            self.assertIsInstance(data_batch, DataBatch)
            self.assertEqual(data_batch.x.shape[0], self.batch_size)
            self.assertEqual(data_batch.y.shape[0], self.batch_size)

    def test_data_loader_random_sampler(self):
        data_loader = DataLoader(
            self.dataset,
            self.batch_size,
            sampler=self.random_sampler,
            collate_fn=self.dataset.collate_fn,
            drop_last=True)

        self.assertEqual(len(list(data_loader)), len(self.dataset) // self.batch_size)
        for data_batch in data_loader:
            self.assertIsInstance(data_batch, DataBatch)
            self.assertEqual(data_batch.x.shape[0], self.batch_size)
            self.assertEqual(data_batch.y.shape[0], self.batch_size)
