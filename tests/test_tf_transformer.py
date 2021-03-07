import torch
import unittest
import tensorflow as tf
import numpy.testing as npt

from transformer.tf.model import attention, MultiHeadAttention

class TestAttention(unittest.TestCase):

    def test_attention(self):
        temp_k = tf.constant([[10,0,0],
                [0,10,0],
                [0,0,10],
                [0,0,10]], dtype=tf.float32)  # (4, 3)

        temp_v = tf.constant([[   1,0],
                            [  10,0],
                            [ 100,5],
                            [1000,6]], dtype=tf.float32)  # (4, 2)

        temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
        out, attn = attention(temp_q, temp_k, temp_v)
        npt.assert_almost_equal(attn.numpy(), [[0., 1., 0., 0.]], 0)
        npt.assert_almost_equal(out.numpy(), [[10., 0.]], 0)

        temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
        out, attn = attention(temp_q, temp_k, temp_v)
        npt.assert_almost_equal(attn.numpy(), [[0.,  0.,  0.5, 0.]], 0)
        npt.assert_almost_equal(out.numpy(), [[550., 5.5]], 0)

        temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
        out, attn = attention(temp_q, temp_k, temp_v)
        npt.assert_almost_equal(attn.numpy(), [[0.5, 0.5, 0., 0.]], 0)
        npt.assert_almost_equal(out.numpy(), [[5.5, 0.]], 0)

class TestMultiHeadedAttention(unittest.TestCase):

    def test_multi_head_attention(self):

        temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
        y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
        out, attn = temp_mha(k=y, q=y, v=y)
        self.assertEqual(out.shape, (1, 60, 512))
        self.assertEqual(attn.shape, (1, 8, 60, 60))


    def test_multi_head_attention_2(self):
        batch_size = 4
        heads = 8
        dims = 256
        query = tf.random.uniform((batch_size, dims))
        key = tf.random.uniform((batch_size, dims))
        value = tf.random.uniform((batch_size, dims))

        temp_mha = MultiHeadAttention(d_model=256, num_heads=8)
        out, attn = temp_mha(q=query, k=key, v=value)
        self.assertEqual(out.numpy().dtype, query.numpy().dtype)
        self.assertEqual(out.shape, (batch_size, 1, heads * dims // heads))


