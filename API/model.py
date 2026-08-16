import sys, os
sys.path.append(os.path.abspath('..'))

import tensorflow as tf
from Transformer.transformer import TransformerNoEnc
from Gumbel.gumbel import GumbelSoftmax
import numpy as np
import selfies as sf
from utils_functions import get_props
from sklearn.metrics import mean_squared_error

class GAN(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.transformer_layer = TransformerNoEnc(
            num_layers=2,
            d_model=64,
            dff=128,
            num_heads=4,
            dropout=0.1,
            seq_len=25,
            vocab_size=29,
        )

        _tmp_input_1 = np.random.randn(1, 23)
        _tmp_input_2 = np.random.randn(1, 1)
        _ = self.transformer_layer((_tmp_input_1, _tmp_input_2))

        self.gumbel_sm_layer = GumbelSoftmax(self.transformer_layer)

        self.inputs_context = tf.keras.layers.Input((23, ))
        self.inputs_dec = tf.keras.layers.Input((None, ))
        self.gumbel_layer_out = self.gumbel_sm_layer((self.inputs_context, self.inputs_dec), 
                                                 beta=1.0)
        
        self.generator = tf.keras.Model(inputs=(self.inputs_context, self.inputs_dec), 
                                        outputs=self.gumbel_layer_out)


def selfies_preprocessing(selfies_set, id_to_token, special_tokens):
	"""Tokenized data to SELFIES"""
	all_smiles = []
	for selfies in selfies_set:
		single_selfies = ""
		for token in selfies:
			curr_token = id_to_token[str(token)]
			if curr_token not in special_tokens:
				single_selfies += curr_token
		all_smiles.append(single_selfies)
	return all_smiles

def get_samples(generator, props, token_to_id, seq_len, start_token='[START]', beta=1.0):
	batch_shape = props.shape[0]
	noise_props = tf.convert_to_tensor(props)

	start_id = token_to_id[start_token]
	dec_tokens = tf.fill([batch_shape, 1], start_id)
	for _ in range(seq_len):
		probs = generator((noise_props, dec_tokens), beta=beta)
		next_logits = probs[:, -1, :]
		next_logits = tf.argmax(next_logits, axis=-1)
		next_tokens = tf.expand_dims(next_logits, axis=1)
		next_tokens = tf.cast(next_tokens, tf.int32)
		dec_tokens = tf.concat([dec_tokens, next_tokens], axis=1)
	
	return dec_tokens

def get_preds(input_props, generator, std_sclr, token_to_id, seq_len, id_to_token, special_tokens):
    # input_props = np.array(input_props, dtype=np.float32).reshape(-1, props_dim)
    input_props = std_sclr.transform(input_props)

    preds = get_samples(generator=generator, props=input_props, token_to_id=token_to_id, seq_len=seq_len, start_token='[START]', beta=1.0)
    preds = np.array(preds)
	
    get_selfies = selfies_preprocessing(preds, id_to_token, special_tokens)
    get_smiles = sf.decoder(get_selfies[0])

    return get_smiles

def get_props_and_error(smiles, props):
    get_add_props = get_props((smiles, ))
	
    get_add_props = np.array(get_add_props)
    props = np.array(props).reshape(1, -1)

    error = mean_squared_error(y_true=props, y_pred=get_add_props)
    return np.squeeze(get_add_props), error