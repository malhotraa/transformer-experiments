import torch
from torch import nn
import unittest

from transformer.model import Encoder, clones, MultiHeadedAttention, attention, EncoderLayer, DecoderLayer, SublayerConnection, PositionwiseFeedForward, Decoder

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 8
        self.heads = 8
        self.dims = 256
        self.dff = 1024
        self.dropout_prob = 0.5
        self.num_enc = 6
        self.num_dec = 6
        self.query = torch.rand(self.batch_size, self.heads * self.seq_len, self.dims)
        self.key = torch.rand(self.batch_size, self.heads * self.seq_len, self.dims)
        self.value = torch.rand(self.batch_size, self.heads * self.seq_len, self.dims)
        self.multi_head_attn = MultiHeadedAttention(self.heads, self.dims)
        self.feed_fwd = PositionwiseFeedForward(self.dims, self.dff, self.dropout_prob)
        self.enc_layer = EncoderLayer(
            self.dims, self.multi_head_attn, self.feed_fwd, self.dropout_prob)
        self.enc = Encoder(self.enc_layer, self.num_enc)
        self.src_mask = torch.rand(self.batch_size, 1, self.heads * self.seq_len)
        self.tgt_mask = torch.rand(self.batch_size, 1, self.heads * self.seq_len)
        self.dec_layer = DecoderLayer(
            self.dims, self.multi_head_attn, self.multi_head_attn, self.feed_fwd, self.dropout_prob
        )

    def test_multi_head_attention(self):
        out = self.multi_head_attn.forward(self.query, self.key, self.value)
        self.assertEqual(out.dtype, self.query.dtype)
        self.assertEqual(out.device, self.query.device)
        self.assertEqual(out.shape, (self.batch_size, self.heads * self.seq_len, self.heads * self.dims // self.heads))

    def test_attention_batch_size_one_single_head(self):
        x = torch.rand(self.seq_len, self.dims)
        out, weights = attention(query=x, key=x, value=x)
        self.assertEqual(out.shape, (self.seq_len, self.dims))
        self.assertEqual(weights.shape, (self.seq_len, self.seq_len))
        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(weights.dtype, x.dtype)
        self.assertEqual(out.device, x.device)
        self.assertEqual(weights.device, x.device)

    def test_attention_batch_size_one_multi_head(self):
        x = torch.rand(self.heads * self.seq_len, self.dims)
        out, weights = attention(query=x, key=x, value=x)
        self.assertEqual(out.shape, (self.heads * self.seq_len, self.dims))
        self.assertEqual(weights.shape, (self.heads * self.seq_len, self.heads * self.seq_len))
        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(weights.dtype, x.dtype)
        self.assertEqual(out.device, x.device)
        self.assertEqual(weights.device, x.device)

    def test_attention_batch_size_many_multi_head(self):
        out, weights = attention(query=self.query, key=self.key, value=self.value)
        self.assertEqual(out.shape, (self.batch_size, self.heads * self.seq_len, self.dims))
        self.assertEqual(weights.shape, (self.batch_size, self.heads * self.seq_len, self.heads * self.seq_len))
        self.assertEqual(out.dtype, self.query.dtype)
        self.assertEqual(weights.dtype, self.query.dtype)
        self.assertEqual(out.device, self.query.device)
        self.assertEqual(weights.device, self.query.device)

    def test_attention_batch_size_many_multi_head_mask(self):
        # TODO: add test with a mask
        pass

    def test_clones(self):
        num_clones = 4
        mod = nn.Linear(self.dims, self.dims)
        out = clones(mod, num_clones)
        self.assertIsInstance(out, nn.ModuleList)
        self.assertEqual(len(out), num_clones)
        for i in range(num_clones):
            self.assertIsInstance(out[i], nn.Linear)

    def test_sublayer_connection(self):
        x = torch.rand(self.dims)
        sublayer = nn.Linear(self.dims, self.dims)
        sublayer_conn = SublayerConnection(self.dims, dropout=0.5)
        out = sublayer_conn.forward(x, sublayer)
        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.device, x.device)

    def test_encoder_layer(self):
        out = self.enc_layer.forward(self.query)
        self.assertEqual(out.dtype, self.query.dtype)
        self.assertEqual(out.device, self.query.device)
        self.assertEqual(out.shape, self.query.shape)

    def test_encoder(self):
        enc = Encoder(self.enc_layer, self.num_enc)
        out = enc.forward(self.query)
        self.assertEqual(out.dtype, self.query.dtype)
        self.assertEqual(out.device, self.query.device)
        self.assertEqual(out.shape, self.query.shape)

    def test_decoder_layer(self):
        memory = self.enc.forward(self.query)
        dec = DecoderLayer(self.dims, self.multi_head_attn, self.multi_head_attn, self.feed_fwd, self.dropout_prob)
        out = dec.forward(self.query, memory, self.src_mask, self.tgt_mask)
        self.assertEqual(out.dtype, self.query.dtype)
        self.assertEqual(out.device, self.query.device)
        self.assertEqual(out.shape, self.query.shape)

    def test_decoder(self):
        dec = Decoder(self.dec_layer, self.num_enc)
        out = dec.forward(self.query, self.enc.forward(self.query), self.src_mask, self.tgt_mask)
        self.assertEqual(out.dtype, self.query.dtype)
        self.assertEqual(out.device, self.query.device)
        self.assertEqual(out.shape, self.query.shape)



