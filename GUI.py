from Transformer.transformer import TransformerNoEnc
from Gumbel.gumbel import GumbelSoftmax
import tensorflow as tf
import json
import numpy as np
import selfies as sf
from tkinter import ttk
import tkinter as tk
import joblib

# Tokenization data loading
with open('Transformer/token_to_id.json', 'r') as f:
    token_to_id = json.load(f)
with open('Transformer/id_to_token.json', 'r') as f:
    id_to_token = json.load(f)

# Model
num_layers = 2
d_model = 64
dff = 128
num_heads = 4
dropout = 0.1
SEQ_LEN = 25
vocab_size = len(token_to_id)

transformer_layer = TransformerNoEnc(
	num_layers=num_layers,
	d_model=d_model,
	num_heads=num_heads,
	dff=dff,
	seq_len=SEQ_LEN,
    vocab_size=vocab_size,
	dropout=dropout
)

# Building
props_dim =  23
tmp1, tmp2 = np.random.randn(1, props_dim), np.random.randn(1, 1)
tmp = transformer_layer((tmp1, tmp2))

gumbel_softmax_layer = GumbelSoftmax(transformer_layer)
tmp = gumbel_softmax_layer((tmp1, tmp2), beta=1)

gen_input_context = tf.keras.layers.Input((props_dim, ))
gen_input_dec = tf.keras.layers.Input((None, ))
gumbel_softmax_layer_out = gumbel_softmax_layer((gen_input_context, gen_input_dec), beta=1)
generator = tf.keras.Model(inputs=(gen_input_context, gen_input_dec), outputs=gumbel_softmax_layer_out)
generator.load_weights('SavedWeights/generator.weights.h5')


def get_samples(model, props, token_to_id, seq_len, start_token='[START]', beta=1.0):
	"""Return GAN's predictions"""
	batch_shape = props.shape[0]
	noise_props = tf.convert_to_tensor(props)

	start_id = token_to_id[start_token]
	dec_tokens = tf.fill([batch_shape, 1], start_id)
	for _ in range(seq_len):
		probs = model((noise_props, dec_tokens), beta=beta)
		next_logits = probs[:, -1, :]
		next_logits = tf.argmax(next_logits, axis=-1)
		next_tokens = tf.expand_dims(next_logits, axis=1)
		next_tokens = tf.cast(next_tokens, tf.int32)
		dec_tokens = tf.concat([dec_tokens, next_tokens], axis=1)
	
	return dec_tokens

special_tokens = ['[PAD]', '[START]', '[END]']
def selfies_preprocessing(selfies_set):
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


# GUI API
root = tk.Tk()
root.title("Sample's Generator")

canvas = tk.Canvas(root)
scrollbar = ttk.Scrollbar(root, command=canvas.yview)
scroll_frame = ttk.Frame(canvas)

scroll_frame.bind('<Configure>', lambda _: canvas.configure(scrollregion=canvas.bbox('all')))
canvas.create_window((0, 0), window=scroll_frame)

canvas.pack(side='left', fill='both', expand=True)
scrollbar.pack(side='right', fill='y')

# Properties names
properties_names = ['A', 'B', 'C', 'Cv', 'G', 'G_atomization', 'H', 'H_atomization', 
					'U', 'U0', 'U0_atomization', 'U_atomization', 'alpha', 
					'gap', 'homo', 'lumo', 'mu', 'r2', 'zpve', 'qed', 'LogP', 'TPSA', 'mw']


# Default properties
defaults = [ 2.64047003e+00,  1.23150003e+00,  9.78320003e-01,  3.37239990e+01,
       -4.60127686e+02, -2.61191893e+00, -4.60083252e+02, -2.85100389e+00,
       -4.60084198e+02, -4.60093384e+02, -2.81628108e+00, -2.83400297e+00,
        7.14800034e+01,  2.99899995e-01, -2.41500005e-01,  5.84000014e-02,
        2.55679989e+00,  1.25091504e+03,  1.57332003e-01,  5.49147801e-01,
       -4.65000000e-01,  3.86900000e+01,  1.30143000e+02]

outputs = []
for idx in range(props_dim):
	label = ttk.Label(scroll_frame, text=properties_names[idx])
	label.grid(row=idx, column=0, padx=10, pady=6, sticky='w')
	
	output = ttk.Entry(scroll_frame, width=50)
	output.grid(row=idx, column=1, padx=10, pady=5)
	output.insert(0, defaults[idx])

	outputs.append(output)
	
# Buttons
button_frame = ttk.Frame(scroll_frame)
button_frame.grid(row=props_dim, column=0, columnspan=2, pady=15)

ttk.Button(button_frame, text='Get predicted SMILES', command=lambda: get_smiles(inputs_props=get_data(), generator=generator, status_label=st_label)).pack(side='left', padx=5)
ttk.Button(button_frame, text='Clear', command=lambda: clear()).pack(side='left', padx=5)

# Data printing
st_label = ttk.Label(scroll_frame, text='Enter data first and wait for results')
st_label.grid(row=30, column=0, columnspan=2, pady=10, padx=10, sticky='w')

def get_data():
	"""Takes data from GUI"""
	return [output.get() for output in outputs]

def clear():
	"""Clear inputs"""
	for output in outputs:
		output.delete(0, tk.END)
	st_label.config(text='Cleared', foreground='gray')

std_sclr = joblib.load('./Transformer/StandardScaler.pkl')
def get_smiles(inputs_props, generator, status_label):
	"""Data pipeline"""
	inputs_props = np.array(inputs_props, dtype=np.float32).reshape(-1, props_dim)
	scaled_props = std_sclr.transform(inputs_props)

	get_preds = get_samples(generator, scaled_props, token_to_id, SEQ_LEN, beta=1.0)
	get_preds = np.array(get_preds)

	get_selfies = selfies_preprocessing(get_preds)	
	get_smiles = sf.decoder(get_selfies[0])

	status_label.config(text=f"SMILES: {get_smiles}", foreground="darkgreen")

root.mainloop()