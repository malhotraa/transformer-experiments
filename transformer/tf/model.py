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