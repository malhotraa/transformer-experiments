import torch
from torch import nn
import unittest

from transformer.pytorch.model import Embeddings, Encoder, clones, MultiHeadedAttention, attention, EncoderLayer, DecoderLayer, SublayerConnection, PositionwiseFeedForward, Decoder, PositionalEncoding, Block, make_gpt

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 16
        self.heads = 8
        self.dims = 256
        self.dff = 1024
        self.dropout_prob = 0.5
        self.num_enc = 6
        self.num_dec = 6
        self.vocab_size = 1024
        self.query = torch.rand(self.batch_size, self.heads, self.seq_len, self.dims)
        self.key = torch.rand(self.batch_size, self.heads, self.seq_len, self.dims)
        self.value = torch.rand(self.batch_size, self.heads, self.seq_len, self.dims)
        self.multi_head_query = torch.rand(self.batch_size, self.seq_len, self.dims)
        self.multi_head_key = torch.rand(self.batch_size, self.seq_len, self.dims)
        self.multi_head_value = torch.rand(self.batch_size, self.seq_len, self.dims)
        self.multi_head_attn = MultiHeadedAttention(self.heads, self.dims)
        self.feed_fwd = PositionwiseFeedForward(self.dims, self.dff, self.dropout_prob)
        self.enc_layer = EncoderLayer(
            self.dims, self.multi_head_attn, self.feed_fwd, self.dropout_prob)
        self.enc = Encoder(self.enc_layer, self.num_enc)
        self.dec_layer = DecoderLayer(
            self.dims, self.multi_head_attn, self.multi_head_attn, self.feed_fwd, self.dropout_prob)
        self.mask = torch.ones((self.batch_size, self.seq_len, self.seq_len), dtype=torch.bool)
        self.mask[: , :, int(0.5 * self.seq_len):] = False

    def test_multi_head_attention(self):
        out = self.multi_head_attn.forward(self.multi_head_query, self.multi_head_key, self.multi_head_value)
        self.assertEqual(out.dtype, self.multi_head_query.dtype)
        self.assertEqual(out.device, self.multi_head_query.device)
        self.assertEqual(out.shape, self.multi_head_query.shape)

    def test_attention_single_head(self):
        x = torch.rand(1, self.seq_len, self.dims)
        out, weights = attention(query=x, key=x, value=x)
        self.assertEqual(out.shape, (1, self.seq_len, self.dims))
        self.assertEqual(weights.shape, (1, self.seq_len, self.seq_len))
        self.assertEqual(out.dtype, x.dtype)
        self.assertEqual(weights.dtype, x.dtype)
        self.assertEqual(out.device, x.device)
        self.assertEqual(weights.device, x.device)

    def test_attention_multi_head(self):
        out, weights = attention(query=self.query, key=self.key, value=self.value)
        self.assertEqual(out.shape, self.query.shape)
        self.assertEqual(weights.shape, (self.batch_size, self.heads, self.seq_len, self.seq_len))
        self.assertEqual(out.dtype, self.query.dtype)
        self.assertEqual(weights.dtype, self.query.dtype)
        self.assertEqual(out.device, self.query.device)
        self.assertEqual(weights.device, self.query.device)

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
        out = self.enc_layer.forward(self.multi_head_query)
        self.assertEqual(out.dtype, self.multi_head_query.dtype)
        self.assertEqual(out.device, self.multi_head_query.device)
        self.assertEqual(out.shape, self.multi_head_query.shape)

    def test_encoder(self):
        enc = Encoder(self.enc_layer, self.num_enc)
        out = enc.forward(self.multi_head_query)
        self.assertEqual(out.dtype, self.multi_head_query.dtype)
        self.assertEqual(out.device, self.multi_head_query.device)
        self.assertEqual(out.shape, self.multi_head_query.shape)

    def test_decoder_layer(self):
        memory = self.enc.forward(self.multi_head_query)
        dec = DecoderLayer(
            self.dims,
            self.multi_head_attn,
            self.multi_head_attn,
            self.feed_fwd,
            self.dropout_prob)
        out = dec.forward(self.multi_head_query, memory, self.mask, self.mask)
        self.assertEqual(out.dtype, self.multi_head_query.dtype)
        self.assertEqual(out.device, self.multi_head_query.device)
        self.assertEqual(out.shape, self.multi_head_query.shape)

    def test_decoder(self):
        dec = Decoder(self.dec_layer, self.num_enc)
        out = dec.forward(self.multi_head_query, self.enc.forward(self.multi_head_query), self.mask, self.mask)
        self.assertEqual(out.dtype, self.multi_head_query.dtype)
        self.assertEqual(out.device, self.multi_head_query.device)
        self.assertEqual(out.shape, self.multi_head_query.shape)

    def test_position_encoding(self):
        positional_enc = PositionalEncoding(self.dims, self.dropout_prob)
        out = positional_enc.forward(self.multi_head_query)
        self.assertEqual(out.dtype, self.multi_head_query.dtype)
        self.assertEqual(out.device, self.multi_head_query.device)
        self.assertEqual(out.shape, self.multi_head_query.shape)

    def test_masked_attention(self):
        out, weights = attention(query=self.multi_head_query, key=self.multi_head_key, value=self.multi_head_value, mask=self.mask)
        self.assertEqual(out.shape, self.multi_head_query.shape)
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len, self.seq_len))
        self.assertEqual(out.dtype, self.multi_head_query.dtype)
        self.assertEqual(weights.dtype, self.multi_head_query.dtype)
        self.assertEqual(out.device, self.multi_head_query.device)
        self.assertEqual(weights.device, self.multi_head_query.device)
        self.assertTrue(torch.allclose(weights[:, :, int(0.5 * self.seq_len):],
                                       torch.zeros(self.batch_size, self.seq_len, int(0.5 * self.seq_len))))

    def test_masked_multi_headed_attention(self):
        out = self.multi_head_attn.forward(self.multi_head_query, self.multi_head_key, self.multi_head_value, self.mask)
        self.assertEqual(out.dtype, self.multi_head_query.dtype)
        self.assertEqual(out.device, self.multi_head_query.device)
        self.assertEqual(out.shape, self.multi_head_query.shape)

    def test_embeddings(self):
        x = torch.randint(10, (self.batch_size, self.seq_len), dtype=torch.long)
        embeddings = Embeddings(self.dims, self.vocab_size)
        out = embeddings.forward(x)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.device, x.device)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.dims))

    def test_block(self):
        block = Block(self.dims, self.heads, self.dropout_prob)
        out = block.forward(self.multi_head_query)
        self.assertEqual(out.shape, self.multi_head_query.shape)
        self.assertEqual(out.dtype, self.multi_head_query.dtype)
        self.assertEqual(out.shape, self.multi_head_query.shape)

    def test_make_gpt(self):
        """
        Notes on shapes:
            input: (batch_size, seq_len)
            output of embeddings: (batch_size, seq_len, d_model)
            output of positional_encoding: (batch_size, seq_len, d_model)
            output of block: (batch_size, seq_len, d_model)
            output of model: (batch_size, seq_len, d_model)

        """
        model = make_gpt(self.vocab_size, self.num_enc, self.dims, self.dff, self.heads, self.dropout_prob)
        x = torch.randint(10, (self.batch_size, self.seq_len), dtype=torch.long)
        out = model(x)
        self.assertEqual(out.shape, self.multi_head_query.shape)
        self.assertEqual(out.dtype, self.multi_head_query.dtype)
        self.assertEqual(out.shape, self.multi_head_query.shape)
