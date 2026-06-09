import tensorflow as tf

def positional_encoding_f(seq_len, d_model):
    pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]

    index = tf.math.floordiv(i, 2)
    denominator = 10_000 ** ((2 * index) / d_model)
    ang = pos / denominator

    sin = tf.sin(ang)
    cos = tf.cos(ang)

    mask = tf.cast(tf.range(d_model) % 2 == 0, tf.float32)[tf.newaxis, :]
    return sin * mask + cos * (1 - mask)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.positional_encoding = positional_encoding_f(seq_len=self.seq_len, d_model=self.d_model)
    
    def call(self, x):
        neded_len = tf.shape(x)[1]

        pos_e = self.positional_encoding[:neded_len, :]
        pos_e = pos_e[tf.newaxis, :, :]
        pos_e = tf.cast(pos_e, x.dtype)
        return x + pos_e


class Projection(tf.keras.layers.Layer):
	def __init__(self, d_model, seq_len):
		super().__init__()
		self.seq_len = seq_len
		self.d_model = d_model
		self.dense = tf.keras.layers.Dense(seq_len * d_model)
	
	def call(self, x):
		x = self.dense(x)
		return tf.reshape(x, (-1, self.seq_len, self.d_model))



class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dropout):
        super().__init__()
        self.muliHeadAttention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)
        self.add_layer = tf.keras.layers.Add()
        self.layerNormalization = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        att = self.muliHeadAttention(query=x, value=x, key=x, use_causal_mask=True)
        x = self.add_layer([x, att])
        x = self.layerNormalization(x)
        return x


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dropout):
        super().__init__()
        self.muliHeadAttention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=dropout)
        self.add_layer = tf.keras.layers.Add()
        self.layerNormalization = tf.keras.layers.LayerNormalization()
    
    def call(self, context, x):
        att = self.muliHeadAttention(query=x, value=context, key=context)
        x = self.add_layer([x, att])
        x = self.layerNormalization(x)
        return x
    

class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout=0.2):
        super().__init__()
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout)
        ])
        self.add_layer = tf.keras.layers.Add()
        self.layerNormalization = tf.keras.layers.LayerNormalization()
    
    def call(self, x):
        x = self.add_layer([x, self.network(x)])
        x = self.layerNormalization(x)
        return x


class DecoderSubblayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout):
        super().__init__()
        self.self_attention = SelfAttention(num_heads=num_heads, d_model=d_model, dropout=dropout)
        self.cross_attention = CrossAttention(num_heads=num_heads, d_model=d_model, dropout=dropout)
        self.feedforward = FeedForwardNetwork(d_model=d_model, dff=dff)

    def call(self, x, context):
        x = self.self_attention(x)
        x = self.cross_attention(x=x, context=context)
        x = self.feedforward(x)
        return x
    

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, seq_len, dropout):
        super().__init__()
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.positional_encoding = PositionalEncoding(seq_len=seq_len, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.dec_layers = []
        for _ in range(num_layers):
            self.dec_layers.append(DecoderSubblayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout=dropout))

    def call(self, x, context):
        x = self.embedding_layer(x)
        x = self.positional_encoding(x)

        for layer in self.dec_layers:
            x = layer(x, context)
        
        return self.dropout(x)
    

class TransformerNoEnc(tf.keras.Model):
    """Modified transformer implementation"""
    def __init__(self, num_layers, num_heads, vocab_size, d_model, dff, seq_len, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.dff = dff 
        self.seq_len = seq_len
        self.dropout = dropout

        self.projection = Projection(d_model=self.d_model, seq_len=self.seq_len)
        self.decoder = Decoder(num_layers=num_layers, num_heads=num_heads, vocab_size=vocab_size, d_model=d_model, dff=dff, seq_len=seq_len, dropout=dropout)
        self.last_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        context, x = inputs
        context = self.projection(context)
        x = self.decoder(x=x, context=context)
        return self.last_layer(x)
    
def loss_fn(y_true, y_pred):
    """Masked crossentropy loss function"""
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.cast(tf.not_equal(y_true, 0), loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)