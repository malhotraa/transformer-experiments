import torch
import unittest

from transformer.model import MultiHeadedAttention

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.heads = 8
        self.dims = 256
        self.query = torch.rand(self.batch_size, self.dims)
        self.key = torch.rand(self.batch_size, self.dims)
        self.value = torch.rand(self.batch_size, self.dims)
        self.multi_head_attn = MultiHeadedAttention(self.heads, self.dims)

    def tearDown(self):
        pass

    def test_multi_head_attention(self):
        out = self.multi_head_attn.forward(self.query, self.key, self.value)
        self.assertEqual(out.dtype, self.query.dtype)
        self.assertEqual(out.shape, (self.batch_size, 1, self.heads * self.dims // self.heads))