import torch
import unittest
import tensorflow as tf
import numpy.testing as npt

from transformer.tf.model import attention

class TestAttention(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


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


