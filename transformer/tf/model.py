import tensorflow as tf

def attention(q, k, v, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"

    dk = tf.cast(tf.shape(k)[-1], tf.float32)

    matmul_qk = tf.matmul(q, k, transpose_b=True)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = tf.transpose(tf.reshape(q, (batch_size, -1, self.num_heads, self.depth)), perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len_q, depth)
        k = tf.transpose(tf.reshape(k, (batch_size, -1, self.num_heads, self.depth)), perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len_q, depth)
        v = tf.transpose(tf.reshape(v, (batch_size, -1, self.num_heads, self.depth)), perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len_q, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
