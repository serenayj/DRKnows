import pickle
import networkx as nx
import json 
from transformers import AutoTokenizer, AutoModel
import numpy as np
import time
import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F
import math 
from tqdm import tqdm 

tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
cui_aui_mappings = pickle.load(open("sm_t047_cui_aui_eng.pkl","rb")) 
cui_vocab = json.load(open("CUI_Vocab.json",'r')) 

device = torch.device('cuda') 

def generate_cui_emb(cuis, model, tokenizer, input_type="Node"):
	"""
	Given a list of CUIs, generate CUI embeddings based on the phrase mappings 
	"""
	cui_phrases = [] 
	for c in cuis:
		text = "[SEP]".join([i[1] for i in cui_aui_mappings[c]]) # concatenating atomic phrases of a CUI with [SEP] 
		cui_phrases.append(text)
	# be careful that the tokenizer has to be batch_encode_plus 
	# set maximum phrases length is 256
	task_tks = tokenizer.batch_encode_plus(cui_phrases, truncation=True, padding=True, max_length=256, return_tensors="pt")

	src_outputs = model(**task_tks.to(device))
	cui_reprs = src_outputs.pooler_output # B X H 
	del src_outputs 
	return cui_reprs 

outs = {}
all_keys = list(cui_vocab.keys()) 
model.to(device)
for k_id in tqdm(range(0, len(all_keys), 1)):
	if k_id +1 < len(all_keys):
		ks  = all_keys[k_id: k_id+1]
	else:
		ks = all_keys[k_id:]
	output = generate_cui_emb(ks, model, tokenizer) 
	for _, k in enumerate(ks):
		outs[k] = output[_,:].unsqueeze(0).to(torch.device('cpu')) 
	del output 

import pickle
with open("SNOMED_CUI_Embedding.pkl","wb") as outf:
	pickle.dump(outs, outf, protocol=pickle.HIGHEST_PROTOCOL)
