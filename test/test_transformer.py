import torch
import unittest

from transformer.model import MultiHeadedAttention, attention

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 8
        self.heads = 8
        self.dims = 256
        self.query = torch.rand(self.batch_size, self.dims)
        self.key = torch.rand(self.batch_size, self.dims)
        self.value = torch.rand(self.batch_size, self.dims)
        self.multi_head_attn = MultiHeadedAttention(self.heads, self.dims)

    def test_multi_head_attention(self):
        out = self.multi_head_attn.forward(self.query, self.key, self.value)
        self.assertEqual(out.dtype, self.query.dtype)
        self.assertEqual(out.device, self.query.device)
        self.assertEqual(out.shape, (self.batch_size, 1, self.heads * self.dims // self.heads))

    def test_attention_batch_size_one_single_head(self):
        y = torch.rand(self.seq_len, self.dims)
        out, weights = attention(query=y, key=y, value=y)
        self.assertEqual(out.shape, (self.seq_len, self.dims))
        self.assertEqual(weights.shape, (self.seq_len, self.seq_len))
        self.assertEqual(out.dtype, y.dtype)
        self.assertEqual(weights.dtype, y.dtype)
        self.assertEqual(out.device, y.device)
        self.assertEqual(weights.device, y.device)

    def test_attention_batch_size_one_multi_head(self):
        y = torch.rand(self.heads, self.seq_len, self.dims)
        out, weights = attention(query=y, key=y, value=y)
        self.assertEqual(out.shape, (self.heads, self.seq_len, self.dims))
        self.assertEqual(weights.shape, (self.heads, self.seq_len, self.seq_len))
        self.assertEqual(out.dtype, y.dtype)
        self.assertEqual(weights.dtype, y.dtype)
        self.assertEqual(out.device, y.device)
        self.assertEqual(weights.device, y.device)

    def test_attention_batch_size_many_multi_head(self):
        y = torch.rand(self.batch_size, self.heads, self.seq_len, self.dims)
        out, weights = attention(query=y, key=y, value=y)
        self.assertEqual(out.shape, (self.batch_size, self.heads, self.seq_len, self.dims))
        self.assertEqual(weights.shape, (self.batch_size, self.heads, self.seq_len, self.seq_len))
        self.assertEqual(out.dtype, y.dtype)
        self.assertEqual(weights.dtype, y.dtype)
        self.assertEqual(out.device, y.device)
        self.assertEqual(weights.device, y.device)

