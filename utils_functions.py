import selfies as sf
import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import DataStructs, QED, Crippen, Descriptors, rdMolDescriptors


def smiles_to_selfies(smiles: str):
	"""Returns SMILES to SELFIES for simple proba"""
	selfie = sf.encoder(smiles)
	return sf.split_selfies(selfie)


def smiles_to_selfies_tokens(smiles, token_to_id, seq_len, pad):
	"""Takes single SELFIES and returns tokenized form"""
	tokens = smiles_to_selfies(smiles)
	token_ids = [token_to_id[token] for token in tokens]
	pad_id = token_to_id[pad]

	if len(token_ids) < seq_len:
		token_ids = token_ids + [pad_id] * (seq_len - len(token_ids))
	else:
		token_ids = token_ids[:seq_len]

	return token_ids


def smiles_to_selfies_tokens_ds(smiles, token_to_id, seq_len, pad):
	"""Takes all SELFIES and returns tokenized form"""
	skipped_mol_n = 0		# Number of skipped molecules
	ae_input = []
	remove_indieces = []	# Indieces to remove, due to conversion error

	for idx, single_smiles in enumerate(smiles):
		try:
			single_ae = smiles_to_selfies_tokens(single_smiles, token_to_id, seq_len, pad)
			ae_input.append(single_ae)
		except:
			remove_indieces.append(idx)
			skipped_mol_n += 1
			continue

	print(f"Number of skipped smiles: {skipped_mol_n}")
	return np.array(ae_input, dtype=np.int32), remove_indieces


def prepare_smiles_to_decoder(single_smiles, token_to_id, seq_len, start, end, pad):
	"""Returns single proba for decoder"""
	tokens = smiles_to_selfies(single_smiles)
	token_ids = (
		[token_to_id[start]] +
		[token_to_id[token] for token in tokens] +
		[token_to_id[end]]
    )
	pad_id = token_to_id[pad]

	if len(token_ids) < seq_len:
		token_ids = token_ids + [pad_id] * (seq_len - len(token_ids))
	else:
		token_ids = token_ids[:seq_len]

	token_ids = np.array(token_ids, dtype=np.int32)
	decoder_in = token_ids[:-1]
	decoder_out = token_ids[1:]

	return decoder_in, decoder_out


def prepare_smiles_to_decoder_ds(smiles, token_to_id, seq_len, start, end, pad):
	"""Takes all smiles and gives to decoder's form"""
	skipped_mol_n = 0		# Number of skipped molecules
	decoder_in_list = []
	decoder_out_list = []
	remove_indieces = []	# Indieces to remove, due to conversion error

	for idx, single_smiles in enumerate(smiles):
		try: 
			single_dec_in, single_dec_out = prepare_smiles_to_decoder(single_smiles, 
															 token_to_id=token_to_id, 
															 seq_len=seq_len, 
															 start=start, 
															 end=end, 
															 pad=pad)
			decoder_in_list.append(single_dec_in)
			decoder_out_list.append(single_dec_out)
		except:
			remove_indieces.append(idx)
			skipped_mol_n += 1
			continue
	
	print(f"Number of skipped smiles: {skipped_mol_n}")
	return (tf.convert_to_tensor(decoder_in_list, dtype=tf.int32),
		 	tf.convert_to_tensor(decoder_out_list, dtype=tf.int32))


def selfies_tokens_to_smiles(tokens, special_tokens_list):
	"""Converts SELFIES to SMILES """
	filtered = [token for token in tokens if token not in special_tokens_list]
	selfies = "".join(filtered)

	try:
		return sf.decoder(selfies)
	except:
		return None


def get_props(smiles_set):
    """Gets additional properties"""
    add_props = []
    for smiles in smiles_set:
        mol = Chem.MolFromSmiles(smiles)

        qed = QED.qed(mol)
        logP = Crippen.MolLogP(mol)
        Ctpsa = rdMolDescriptors.CalcTPSA(mol)
        mw = Descriptors.MolWt(mol)
        
        add_props.append([qed, logP, Ctpsa, mw])
    return add_props