### Gradient accmulation enabling 
### changes in V4: 
### 1. The flow of Graph Reasoning and Representation is changed: new flow: Representation learning -> Reasoning 
### 2. Enabling intermediate loss computation (as in V1)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os 
import numpy as np
import time
import torch
from torch.autograd import Variable 
import torch.nn as nn
import torch.nn.functional as F
import math 
from transformers import AutoTokenizer, AutoModel
import pandas as pd 
import json
from tqdm import tqdm 
import argparse
import logging
import datetime
import random 
import re
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pickle   
import networkx as nx
#from quickumls import *
#import faiss 
# from mpi4py import MPI # for parallelization


tokenizer = AutoTokenizer.from_pretrained("/home/ygao/LAB_SHARED/home/ygao/nlp_models/SapBERT-from-PubMedBERT-fulltext")
model = AutoModel.from_pretrained("/home/ygao/LAB_SHARED/home/ygao/nlp_models/SapBERT-from-PubMedBERT-fulltext")

import pandas as pd 
from torch.nn.utils.rnn import pad_sequence

import pickle
from collections import OrderedDict

torch.manual_seed(2023)
random.seed(2023) 

# sync_networks across the different cores
# def sync_networks(network):
#     """
#     netowrk is the network you want to sync
#     """
#     comm = MPI.COMM_WORLD
#     flat_params = _get_flat_params_or_grads(network, mode='params')
#     comm.Bcast(flat_params, root=0)
#     # set the flat params back to the network
#     _set_flat_params_or_grads(network, flat_params, mode='params')

# # sync the grads across the different cores
# def sync_grads(network):
#     flat_grads = _get_flat_params_or_grads1(network, mode='grads')
#     comm = MPI.COMM_WORLD
#     global_grads = np.zeros_like(flat_grads)
#     comm.Allreduce(flat_grads, global_grads, op=MPI.SUM)
#     _set_flat_params_or_grads1(network, global_grads, mode='grads')

# # get the flat grads and ignore None Type grads
# def _get_flat_params_or_grads1(network, mode='params'):
#     """
#     include two kinds: grads and params
#     """
#     attr = 'data' if mode == 'params' else 'grad'
#     grads = []
#     for param in network:
#         if param.grad is not None:
#             grads.append(param.grad.cpu().numpy().flatten())
#     return np.concatenate(grads)
#     #return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network])

# # set the flat grads and ignore None Type grads
# def _set_flat_params_or_grads1(network, flat_params, mode='params'):
#     """
#     include two kinds: grads and params
#     """
#     attr = 'data' if mode == 'params' else 'grad'
#     # the pointer
#     pointer = 0
#     for param in network:
#         #getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
#         #pointer += param.data.numel()
#         if param.grad is not None:
#             param.grad.copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
#             pointer += param.data.numel()

# # get the flat params
# def _get_flat_params_or_grads(network, mode='params'):
#     """
#     include two kinds: grads and params
#     """
#     attr = 'data' if mode == 'params' else 'grad'
#     return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])

# # set the flat params
# def _set_flat_params_or_grads(network, flat_params, mode='params'):
#     """
#     include two kinds: grads and params
#     """
#     attr = 'data' if mode == 'params' else 'grad'
#     # the pointer
#     pointer = 0
#     for param in network.parameters():
#         getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
#         pointer += param.data.numel()

def collate_fn(data):
	input_cui_tks, input_text_tks, cuis_ids, golds, k2_golds, k3_golds,  = zip(*data)
	input_cui_tks_ids, input_cui_tks_attn = [], [] 
	input_text_tks_ids, input_text_tks_attn = [], [] 

	for i in range(len(data)):
		input_cui_tks_ids.append(input_cui_tks[i]["input_ids"])
		input_cui_tks_attn.append(input_cui_tks[i]["attention_mask"])
		input_text_tks_ids.append(input_text_tks[i]["input_ids"])
		input_text_tks_attn.append(input_text_tks[i]["attention_mask"])

	rst_cui_tks_ids = pad_sequence(input_cui_tks_ids)
	rst_cui_tks_attn = pad_sequence(input_cui_tks_attn)
	rst_text_tks_ids = pad_sequence(input_text_tks_ids)
	rst_text_tks_attn = pad_sequence(input_text_tks_attn)

	return (rst_cui_tks_ids, rst_cui_tks_attn, rst_text_tks_ids, rst_text_tks_attn, cuis_ids, golds, k2_golds, k3_golds)

class PretrainData(Dataset):
	""" files in json format 
	"""
	def __init__(self, datapath, tokenizer, cui_vocab, cui_flag=False,intermediate=True,oracle=False,k=None):
		self.data = json.load(open(datapath,'r')) 
		self.all_keys = list(self.data.keys())
		self.train_keys = self.all_keys[:int(len(self.all_keys)*0.15)]
		self.dev_keys = self.all_keys[int(len(self.all_keys)*0.15):]
		self.tokenizer = tokenizer 
		self.cui_vocab = cui_vocab 
		self.intermediate = intermediate
		self.cui_flag = cui_flag
		self.k = k # 2 or 3 if not None 
		self.oracle = oracle 
		# self.index_dict = [i for i in range(len(self.data))]
		# mpi_id = MPI.COMM_WORLD.Get_rank() # get the mpi id
		# interval = len(self.data) // MPI.COMM_WORLD.Get_size() # get the interval to split the data
		# # print(interval)    
		# self.index_dict = self.index_dict[mpi_id*interval: mpi_id*interval + interval] # split the data
		# print('original size: {}'.format(len(self.data)))
		# print(len(self.index_dict))
		# raise NotImplementedError

	def __len__(self):
		return len(self.data)
		# return len(self.index_dict) # return the length of the data


	def __getitem__(self, index):
		k = self.all_keys[index]
		# k = self.all_keys[self.index_dict[index]] # get the key
		text = self.data[k]['input text'] # input text 
		# place holder for input context
		cuis_ids = [self.cui_vocab[i] for i in self.data[k]['input A CUI']] # input candidate cuis
		paths = self.data[k]['paths']
		if self.cui_flag and 'input context' in self.data[k]:
			concept = " [SEP] ".join(list(set(self.data[k]['input context'])))
		else:
			concept = text 
		intermediate_golds = [] 
		k2_golds = []
		k3_golds = [] 
		input_cui_tks = self.tokenizer(concept, 
									truncation=True, 
									padding="max_length", 
									max_length=256, 
									pad_to_max_length=True,
									return_tensors="pt")
		input_text_tks = self.tokenizer(text, 
										truncation=True, 
										padding="max_length",
										max_length=256, 
										pad_to_max_length=True,
										return_tensors="pt")

		golds = [] 
		golds = [self.cui_vocab[p[-1]] for p in paths if self.cui_vocab[p[-1]] not in golds]
		if self.oracle: # if in a oracel exp setting 
			starting_nodes = [] 
			starting_nodes = [self.cui_vocab[p[0]] for p in paths if self.cui_vocab[p[0]] not in starting_nodes]
			cuis_ids = starting_nodes 

		return input_cui_tks, input_text_tks, cuis_ids, golds, k2_golds, k3_golds 


# ====================== gnn_utils ===================
# Graph utils functions 
def retrieve_cuis(text,g, matcher):
	# Retrieve cuis from quickUMLS 
	output = matcher.match(text)
	#output
	cui_output= [ii['cui'] for i in output for ii in i if ii['cui'] in g.nodes]
	terms = [ii['term'] for i in output for ii in i if ii['cui'] in g.nodes]
	cui_outputs = set(cui_output)

	# answer: C0010346 
	return cui_outputs, output

def retrieve_subgraphs(cuis, g):
	# Get subgraphs into a dictionary 
	paths = {}
	for c in cuis:
		paths[c] = [] 
		nodes = list(g.neighbors(c))
		for n in nodes:
			edge_label = g.get_edge_data(c, n)['label']
			paths[c].append([n, edge_label])
	return paths 


def retrieve_phrases(paths, cui_aui_mappings):
	# Map CUI back to phrases to get representation  
	phrase_paths = {}
	for s, t in paths.items():
		sp = cui_aui_mappings[s][0][1] 
		phrase_paths[sp] = []
		for tn in t:
			vp = cui_aui_mappings[tn[0]][0][1]
			phrase_paths[sp].append([vp, tn[1]])
	return phrase_paths



def retrieve_neighbors_paths_no_self(cui_lists, g, prev_candidate_paths_df):
	#import queue 
	"""Important function to reformat paths and direct neighbors 
	Input: 
	cui_lists: a list of CUIs that start searching
	g: current graph 
	candidate_paths_df: if not None, it is the history one-hot path from previous traversal iteration
	Output:
	all_paths: a list of one-hot hop given cui_lists 
	all_neighbors: a list of concepts that will be the candidate predictions
	path_memories: a list of list with four elements: visited source nodes from the prev iteration; 
													  starting nodes at the current iteration (aka cui_list);
													  current candidate node, 
													  current candidate edge 
	"""
	cui_neighbors = retrieve_subgraphs(cui_lists, g) # dictionary of cuis and their neighrbos 
	all_neighbors = [] 
	all_paths = [] 
	path_memories = [] # dict or list? 
	path_buffer = {} # path buffer, a list of dictionary indicating what sources lead to the current target
	if prev_candidate_paths_df is None: 
		all_neighbors = [vv[0] for k,v in cui_neighbors.items() for vv in v if len(v) !=0] # list of neighbor nodes 
		all_paths = [[k, vv[0], vv[1]] for k,v in cui_neighbors.items() for vv in v if len(v) !=0] # list of one-hop path
		path_memories = [[[k], k, vv[0], vv[1]] for k,v in cui_neighbors.items() for vv in v if len(v) !=0]
	else:
		# faster version using itertuples 
		for _ in  prev_candidate_paths_df.itertuples():
			src, tgt = _.Src, _.Tgt
			if src == tgt:
				continue 
			if tgt in path_buffer:
				path_buffer[tgt].append(src)
			else:
				path_buffer[tgt]= [src]
		# remove a specific path where it is the self edge at the first hop 
		for k,v in cui_neighbors.items():
			#print("path buffer k {path_buffer[k]} given k", )
			if len(v) == 0:
				continue
			if k not in path_buffer:
				record = [k,k,k,"self"]
				if record not in path_memories:
					path_memories.append(record)
				continue 
			for vv in v:
				path_memories.append([path_buffer[k], k, vv[0], vv[1]]) 

		all_paths =[pm[1:] for pm in path_memories] 
		all_neighbors =[pm[-2] for pm in path_memories]
	#print("ALL NEIGHRBORS", all_neighbors)
	#print("PATH MEMO: ", path_memories)
	return all_paths, all_neighbors, path_memories  



# Graph retriever utils 
def project_cui_to_vocab(all_paths_df, cui_vocab):
	vocab_idx = []
	new_srcs = all_paths_df['Tgt']
	for _ in new_srcs:
		vocab_idx.append(cui_vocab[_])
	return vocab_idx 


def sort_visited_paths(indices, all_paths_df, visited_path_embs, prev_visited_paths):
	# Postprocess for top-n selected CUIs
	visited_paths = {}
	new_src_cuis_emb = {}
	if len(prev_visited_paths) == 0:
		for _ in indices:
			k = _[0].item() 
			new_src = all_paths_df.iloc[k]['Tgt'] 
			p = all_paths_df.iloc[k]['Src'] + " --> " + all_paths_df.iloc[k]['Edge'] + " --> " + new_src
			visited_paths[new_src] = p # for explainability
			new_src_cuis_emb[new_src] = visited_path_embs[_[0],:] # src CUI embedding to compute next iteration paths
	else:
		for _ in indices:
			k = _[0].item() # index of the top-n path 
			new_src = all_paths_df.iloc[k]['Tgt'] 
			if all_paths_df.iloc[k]['Src'] in prev_visited_paths:
				prev_p = prev_visited_paths[all_paths_df.iloc[k]['Src']]
				p = prev_p +" --> " + all_paths_df.iloc[k]['Edge'] + " --> " + new_src 
			else:
				p = all_paths_df.iloc[k]['Src'] + " --> " + all_paths_df.iloc[k]['Edge'] + " --> " + new_src
			visited_paths[new_src] = p # for explainability
			new_src_cuis_emb[new_src] = visited_path_embs[_[0],:] 

	return visited_paths, new_src_cuis_emb 

def prune_paths(input_text_vec, cand_neighbors_vs, cand_neighbors_list, threshold=0.8):
	"""Purpose: filter out the target CUIs that are not 
	"""
	orig_index = len(cand_neighbors_list) 
	tgt_embs = cand_neighbors_vs.detach().numpy()
	xq = input_text_vec.clone().cpu().detach().numpy() # clone the task embedding 
	new_cand_neighbors_lists = [] 
	d = tgt_embs.shape[-1]
	nb = tgt_embs.shape[0]
	nq = 1
	k =int(nb*threshold) # sample top K nodes with similarity 
	#index = faiss.IndexFlatL2(d)   # build the index for euclidean distance 
	index=faiss.IndexFlatIP(d)     # build the index for cosine distance 
	index.add(tgt_embs)                  # add vectors to the index
	D, I = index.search(xq, k)     # actual search, return distance and index 
	new_cand_neighbor_vs = []
	I_sorted = np.sort(I, axis=1)
	new_cand_neighbor_vs = tgt_embs[I_sorted[0]]
	#print(new_cand_neighbor_vs.shape)
	new_cand_neighbors_lists = [cand_neighbors_list[_] for _ in I_sorted[0]]

	return new_cand_neighbors_lists, new_cand_neighbor_vs

# ====================== gnn  ===================

class CuiEmbedding(object):
	"""
	Backpropagated NOT required, dictionary look-up layer  
	This module could be used for CUI embedding (loaded from pre-trained SAPBERT vectors)
	Module has been tested
	Need to rewrite to read from existing embeddings 
	"""
	def __init__(self, embedding_file):
		super(CuiEmbedding, self).__init__() 
		self.data = pickle.load(open(embedding_file, 'rb')) 

	def encode(self, cui_lists):
		outputs = [] 
		outputs = [torch.as_tensor(self.data[c]) for c in cui_lists] 
		return torch.stack(outputs).squeeze(1)

	def update(self, cui_idx_dicts, cui_embeddings):
		for _, c in enumerate(cui_idx_dicts): # c: CUI, v: index in embedding lookup layer 
			self.data[c] = cui_embeddings[_,:].unsqueeze(0).detach().cpu() # something like this 


class EdgeOneHot(object):
	"""Construct two-layer MLP-type aggreator for GIN model"""
	def __init__(self, edge_mappings):
		super().__init__()
		self.edge_mappings = edge_mappings
		self.onehot_mat = F.one_hot(torch.arange(0, len(self.edge_mappings)), num_classes=len(self.edge_mappings))

	def Lookup(self, edge_lists):
		indices = torch.tensor([self.edge_mappings[e] for e in edge_lists])
		vectors = self.onehot_mat[indices]

		return vectors


class MLP(nn.Module):
	"""Construct two-layer MLP-type aggreator for GIN model"""
	def __init__(self, input_dim, hidden_dim, output_dim):
		super().__init__()
		self.linears = nn.ModuleList()
		# two-layer MLP
		self.mlp_layer1 = nn.Linear(input_dim, hidden_dim, bias=False) 
		self.mlp_layer2 = nn.Linear(hidden_dim, output_dim, bias=False) 
		nn.init.xavier_uniform_(self.mlp_layer1.weight)
		nn.init.xavier_uniform_(self.mlp_layer2.weight)
		self.linears.append(self.mlp_layer1)
		self.linears.append(self.mlp_layer2)

		self.batch_norm = nn.BatchNorm1d((hidden_dim))

	def forward(self, x):
		h = x
		if h.shape[0] == 1: # if only one example, dont use batchnorm 
			h = F.relu(self.linears[0](h), inplace=True) 
		else:
			h = F.relu(self.batch_norm(self.linears[0](h)),inplace=True)
		#print("MLP layer batch norm: ", h)
		return self.linears[1](h)

class GINStack(nn.Module):
	"""
	Stacking NodeAggregateGIN
	"""
	def __init__(self, dim_h, device):
		super().__init__()
		self.conv1 = NodeAggregateGIN(dim_h, dim_h, dim_h, device)
		self.conv2 = NodeAggregateGIN(dim_h, dim_h, dim_h, device)
		self.conv3 = NodeAggregateGIN(dim_h, dim_h, dim_h, device)
		self.lin1 = nn.Linear(dim_h*3, dim_h)
		self.lin2 = nn.Linear(dim_h, dim_h)

	def forward(self, paths_srcs, path_tgt_edges_per_src, candidate_paths_df):
		h1, src_dicts1 = self.conv1(paths_srcs, path_tgt_edges_per_src, candidate_paths_df)
		h2, src_dicts2 = self.conv2(paths_srcs, path_tgt_edges_per_src, candidate_paths_df)
		h3, src_dicts3 = self.conv3(paths_srcs, path_tgt_edges_per_src, candidate_paths_df)

		h = torch.cat((h1,h2,h3), dim=1)

		h = self.lin1(h)
		h = h.relu()
		h = F.dropout(h, p=0.5)
		h= self.lin2(h)
		self.src_df_dicts = self.conv1.src_df_dicts 

		return h, src_dicts1 

class NodeAggregateGIN(nn.Module):
	"""On-the-fly neighboring aggregation for candidate nodes
	Source: MLP-Based Graph Isomorphism (Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (ICLR 2018). 
	"How powerful are graph neural networks?")
	Graph Isomorphism Network with Edge Features, introduced by
	`Strategies for Pre-training Graph Neural Networks <https://arxiv.org/abs/1905.12265>
	h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
		\sum_{j\in\mathcal{N}(i)}\mathrm{ReLU}(h_j^{l} + e_{j,i}^{l})\right)
	where :math:`e_{j,i}^{l}` is the edge feature.

	"""
	def __init__(self, input_dim, hidden_dim, output_dim, device, init_eps=0, learn_eps=False):
		super().__init__()
		self.edge_linear = nn.Linear(hidden_dim+108,hidden_dim)
		self.aggr = MLP(input_dim, hidden_dim, output_dim) 
		self.device = device 
		# to specify whether eps is trainable or not.
		if learn_eps:
			self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
		else:
			self.register_buffer("eps", torch.FloatTensor([init_eps]))


	def message(self, path_tgt_edges_per_src, edge_dicts):
		# \sum_{j\in\mathcal{N}(i)}\mathrm{ReLU}(h_j^{l} + e_{j,i}^{l})\right) 
		msgs = F.relu(path_tgt_edges_per_src)  # same dimensionality as h_n (node embedding, which is 768)
		msgs_dict = {}
		for k,v in edge_dicts.items():
			indices = torch.tensor(v).to(self.device)
			# indices = indices.to(torch.device('cuda'))
			msgs_dict[k] = torch.sum(msgs[indices], dim=0).unsqueeze(0)

		return msgs_dict 

	def organize_neighbors(self, candidate_paths_df):
		r"""Return two dictionaries to help organize the paths and embeddings:
		outputs: dictionary where key is the source node, values are the neighboring nodes and edges 
		src_dicts: dictionary where key is the source node, values is the (start) index of the source node in the df  
		"""
		outputs = {}
		src_dicts = OrderedDict()
		# convert df (all paths) to dict structure where key is the source node, values are the neighboring nodes and edges 
		for rowid, item in candidate_paths_df.iterrows():
			src = item[0]
			if src in outputs:
				outputs[src].append(rowid)
			else:
				outputs[src] = [rowid]
				src_dicts[src] = [rowid] 
		return outputs, src_dicts  

	def forward(self, paths_srcs, path_tgt_edges_per_src, candidate_paths_df):
		#output = self.aggr(node_repr)   
		df_edge_dicts, src_dicts = self.organize_neighbors(candidate_paths_df)
		cand_cuis_mappings = {k: v for v, k in enumerate(set(candidate_paths_df['Src'].to_list()))}
		sorted_cand_cuis_mappings_keys = sorted(list(cand_cuis_mappings.keys()))
		for v,k in enumerate(sorted_cand_cuis_mappings_keys):
			cand_cuis_mappings[k] = v  

		self.src_dicts = src_dicts # for debugging purporse: {CUI: index in path dataframe}
		self.src_df_dicts = df_edge_dicts # to compute CL
		# updated msg
		msgs_dict = self.message(path_tgt_edges_per_src, df_edge_dicts)
		outputs = [] 
		new_src_dicts = {} 
		count = 0
		for k,v in src_dicts.items():
			new_src_dicts[k] = count
			count += 1
			h_src = paths_srcs[cand_cuis_mappings[k]].unsqueeze(0)
			#h_src = paths_srcs[torch.tensor(v[0]).to(self.device)] # h_src original embedding
			h_msg = self.edge_linear(msgs_dict[k].to(self.device)) 
			h_n_prime = (1 + self.eps) * h_src + h_msg 
			outputs.append(h_n_prime)
		raw_feats = torch.cat(outputs) 
		#print(f"Raw Features {raw_feats.shape}")
		output = self.aggr(raw_feats.squeeze(1))
		return output, new_src_dicts  


class PathEncoder(nn.Module):
	"""
	Generate path embedding given src node emb and (target + edge) embedding 
	module has been tested 
	"""
	def __init__(self, hdim, path_dim):
		super(PathEncoder, self).__init__()
		self.d = hdim 
		self.src_weights = nn.Linear(hdim, hdim)
		self.tgt_weights = nn.Linear(path_dim, hdim)
		self.batch_norm = nn.BatchNorm1d((hdim))

		nn.init.xavier_uniform_(self.src_weights.weight)
		nn.init.xavier_uniform_(self.tgt_weights.weight)

	def forward(self, src, tgt):
		#print("SRC weight update"torch.sum)
		hpath = self.src_weights(src) + self.tgt_weights(tgt)
		if hpath.shape[0] == 1:
			hpath = F.relu(hpath, inplace=True)
		else:
			hpath = F.relu(self.batch_norm(hpath), inplace=True)
		return hpath # B X D

class PathEncoderTransformer(nn.Module):
	"""
	Generate path embedding given src node emb and (target + edge) embedding 
	module has been tested 
	"""
	def __init__(self, hdim, path_dim):
		super().__init__()
		self.d = hdim 
		#self.src_weights = nn.Linear(hdim, hdim)
		self.tgt_transform = nn.Linear(path_dim, hdim) # input is target+edge, output is hdim 
		nn.init.xavier_uniform_(self.tgt_transform.weight)

		self.path_encoder = nn.Transformer(d_model=hdim,
										   nhead=3,
										   num_encoder_layers=1,
										   num_decoder_layers=1,
																				   dim_feedforward=128,
										  batch_first=True) 

	def forward(self, src, tgt):
		# input src: a list of source nodes 
		htgt = self.tgt_transform(tgt) # output is B x 768 paths, where B is batch size 
		htgt = htgt.view(htgt.shape[0], 1, htgt.shape[-1]) # reshape to B X 1 X 768
		#print("HTGT shape", htgt.shape)
		#print("SRC SHAPE", src.shape) # expected B X L X 768
		hpath = self.path_encoder(src, htgt) 

		return hpath # B X D


class PathRanker(nn.Module):
	"""
	Input: task embedding, cui embedding, and path embedding 
	Step 1: compute task relevancy and context relevancy 
	Step 2: compute attention scores based on task rel and context rel
	Module has been tested ok; Note that the return shape is B X 4*hdim 
	"""
	def __init__(self, hdim, nums_of_head, attn_weight_mode="Linear", cui_flag=True):
		super(PathRanker, self).__init__()
		self.attention = nn.MultiheadAttention(4*hdim, nums_of_head)
		self.cui_flag = cui_flag
		self.attn_mode = attn_weight_mode
		self.mid_layer = nn.Linear(4*hdim, hdim)
		self.score = nn.Linear(hdim, 1)

		nn.init.xavier_uniform_(self.mid_layer.weight)
		nn.init.xavier_uniform_(self.score.weight)


	def forward(self, task_inputs, cui_inputs, path_embeddings):
		# Infersent based Task relevancy: input text (premise) and paths (hypothesis) 
		task_rel = torch.cat((task_inputs, 
							  path_embeddings, 
							  torch.abs(task_inputs - path_embeddings),
						  task_inputs * path_embeddings), 1)
		if self.cui_flag: # if also computing cui relevancy 
			cui_rel = torch.cat((cui_inputs, 
								 path_embeddings, 
								 torch.abs(cui_inputs - path_embeddings),
						  cui_inputs * path_embeddings), 1)
			#self.merge_repr = task_rel * cui_rel # Hadamard Product of the two matrices
			merge_repr = task_rel * cui_rel 
			attn_output, attn_output_weights = self.attention(merge_repr, merge_repr, merge_repr)
			self.attn_output_weights = attn_output_weights
		else:
			attn_output, attn_output_weights = self.attention(task_rel, task_rel, task_rel)

		scores = self.score(F.relu(self.mid_layer(attn_output)))

		return scores, attn_output, attn_output_weights # attn_output: weighted attention scores, B X 3072 ; attention output weights on scores 

# ====================== model  ===================

class GraphModel(nn.Module):
	def __init__(self, 
				 g, 
				 cui_embedding, # cui_objects
				 hdim, 
				 nums_of_head, 
				 edges_dicts, 
				 cui_aui_mappings, 
				 nums_of_hops,
				 top_n, 
				 device, 
				 gnn_update=False, 
				 cui_flag=True,
				 path_encoder_type="Transformer",
				 gnn_type="Stack",
				 prune_thsh=0.8):
		super(GraphModel, self).__init__()
		#self.n_encoder = NodeEmbedding(tokenizer, model, cui_aui_mappings)
		self.n_encoder = cui_embedding
		self.e_encoder = EdgeOneHot(edges_dicts) # edge_dicts: key: edge name, val: index 
		self.p_encoder_type = path_encoder_type
		if self.p_encoder_type == "Transformer":
			self.p_encoder = PathEncoderTransformer(hdim, hdim+108)
		else:
			self.p_encoder = PathEncoder(hdim, hdim+108)
		self.p_ranker = PathRanker(hdim, nums_of_head, cui_flag)
		#self.gnn = GINE
		self.g = g # network object 
		self.k = nums_of_hops # max k hops 
		self.path_per_batch_size = 128  
		self.top_n = top_n 
		self.logit_loss_mode = "last" # only backpropagate the last selection prob distributions or the entire chain 

		self.edges_mappings = edges_dicts
		self.visited_paths = {} # key: last visited CUI, val: path leading to that CUI 
		self.src_cui_emb = {} # key: last visited CUI, val: embedding of the paths leading to that CUI 
		self.device = device 
		self.candidate_paths_df = None # initialize as None; running path recorder 
		self.gnn_update = gnn_update 
		self.prune_thsh = prune_thsh
		self.gnn_type = gnn_type
		if self.gnn_update: 
			if self.gnn_type == "Stack":
				self.gnn = GINStack(hdim, self.device) 
			else:
				self.gnn = NodeAggregateGIN(hdim, hdim, hdim, self.device)

	def one_iteration(self, task_emb, cui_lists, running_k, context_emb=None, stop_flag=False,prune_thsh=0.8):
		candidate_paths, candidate_neighbors, path_memories= retrieve_neighbors_paths_no_self(cui_lists, self.g, self.candidate_paths_df)  
		candidate_paths_df = pd.DataFrame(candidate_paths, columns=['Src', 'Tgt', 'Edge'])
		path_mem_df = pd.DataFrame(path_memories, columns=['Prev','Src', 'Tgt', 'Edge'])
		self.candidate_paths_df = candidate_paths_df
		#context_emb=None # set context emb as none 

		"""
		Restructure paths to save mem; remove cuis with empty paths; generate cui embedding only once 
		"""
		cand_neighbors_mappings = {k: v for v, k in enumerate(set(candidate_neighbors))} # tgt index set; {taget cui: idx}  
		cand_cuis_mappings = {k: v for v, k in enumerate(set(candidate_paths_df['Src'].to_list()))} # {src cui: idx}

		if len(list(cand_neighbors_mappings.keys())) == 0:
			return [], [], [], True  
		else:
			# fixed the order of each keys to make the test reproduce same result every time
			sorted_cand_neighbors_mappings_keys = sorted(list(cand_neighbors_mappings.keys()))
			for v, k in enumerate(sorted_cand_neighbors_mappings_keys):
				cand_neighbors_mappings[k] = v

			sorted_cand_cuis_mappings_keys = sorted(list(cand_cuis_mappings.keys()))
			for v,k in enumerate(sorted_cand_cuis_mappings_keys):
				cand_cuis_mappings[k] = v

			########
			cand_neighbors_vs = self.n_encoder.encode(sorted_cand_neighbors_mappings_keys)

			if running_k >0: # if not the first iteration
				cand_cui_vs = torch.stack([self.src_cui_emb[v] for v in sorted_cand_cuis_mappings_keys])
			else:
				cand_cui_vs = self.n_encoder.encode(sorted_cand_cuis_mappings_keys).to(self.device)

			prev_srcs = [] 
			if self.p_encoder_type =="Transformer":
				for k,v in cand_cuis_mappings.items(): # k: source CUI, v: index in the df 
					prev_src_repr = self.n_encoder.encode(path_mem_df.iloc[v]['Prev']).to(self.device) # seq len X 768 
					# Concatenating prev source with the current source; 
					if len(cand_cui_vs[v].shape) == 1:
						cui_vs_repr = torch.cat((prev_src_repr,cand_cui_vs[v].unsqueeze(0)), dim=0) # should be (len(prev src) + 1) x 768
					else:
						cui_vs_repr = torch.cat((prev_src_repr,cand_cui_vs[v]), dim=0)
					prev_srcs.append(cui_vs_repr)

				prev_srcs_padded = pad_sequence(prev_srcs, batch_first=True)
				#print("prev src padded shape", prev_srcs_padded.shape)

			all_paths_src, all_paths_tgt_edges = [], [] 

			# Generate (e,v) embedding 
			for i in range(len(candidate_paths_df)):
				v_emb = cand_neighbors_vs[cand_neighbors_mappings[candidate_paths_df.iloc[i]['Tgt']]].unsqueeze(0)
				e_emb = self.e_encoder.onehot_mat[self.edges_mappings[candidate_paths_df.iloc[i]['Edge']]].unsqueeze(0)
				path_emb = torch.cat((v_emb,e_emb),dim=-1)  #new path embs
				all_paths_tgt_edges.append(path_emb)

			paths_tgt_edges = torch.stack(all_paths_tgt_edges)
			
			# ==== Graph representation learning: Update h_v to be contextualized graph representation ====
			if self.p_encoder_type == "Transformer":
				all_paths_src = [prev_srcs_padded[cand_cuis_mappings[candidate_paths_df.iloc[i]['Src']]] for i in range(len(candidate_paths_df))]
			elif not self.gnn_update:
				all_paths_src=[cand_cui_vs[cand_cuis_mappings[candidate_paths_df.iloc[i]['Src']]] for i in range(len(candidate_paths_df))]
			else:
				h_gnn_outputs, updated_node_dicts = self.gnn(cand_cui_vs, paths_tgt_edges, candidate_paths_df) # h_gnn_outputs: num of src x hdim; updated_node_dicts: CUI: idx at candidate_paths_df
				# update_cui_idx_in_hgnn = {cui_v:k_idx for k_idx, cui_v in enumerate(list(updated_node_dicts.keys()))}
				update_cui_idx_in_hgnn = updated_node_dicts
				all_paths_src=[h_gnn_outputs[update_cui_idx_in_hgnn[candidate_paths_df.iloc[i]['Src']]] for i in range(len(candidate_paths_df))] 
				
			paths_srcs = torch.stack(all_paths_src) 
			
			# Start to rank the paths based on inference 
			B = paths_tgt_edges.shape[0] # paths batch size 
			path_scores= [] 
			visited_path_embs = []
			all_indices = torch.arange(0, B).long()
			
			# ==== Graph reasoning: path encoding and ranking ====
			for i in range(0, B, self.path_per_batch_size):
				indices = all_indices[i:i+self.path_per_batch_size]
				src_embs = paths_srcs[indices].to(self.device)
				path_embs = paths_tgt_edges[indices].squeeze(1).to(self.device)
				path_h = self.p_encoder(src_embs, path_embs)
				visited_path_embs.append(path_h)
				exp_task_emb = task_emb.expand(path_h.shape[0], 768)
				if context_emb is not None:
					exp_context_emb = context_emb.expand(path_h.shape[0], 768) 
					scores, attn, attn_weights = self.p_ranker(exp_task_emb, exp_context_emb, path_h.squeeze(1))
				else:
					scores, attn, attn_weights = self.p_ranker(exp_task_emb, exp_task_emb, path_h.squeeze(1))
				path_scores.append(scores)

			visited_path_embs = torch.cat(visited_path_embs,dim=0) 
			
			# ==== Graph post-selection: selecting top-N CUIs for next ====
			final_scores = torch.cat(path_scores, dim=0) # scores on path 
			prev_visited_paths = self.visited_paths 
			if final_scores.shape[0] < self.top_n:
				vals, pred_indices = torch.topk(final_scores, final_scores.shape[0], dim=0) 
			else:
				vals, pred_indices = torch.topk(final_scores, self.top_n, dim=0) # top n 
			visited_paths, new_src_cuis_emb = sort_visited_paths(pred_indices, 
																 candidate_paths_df, 
																 visited_path_embs, 
																 prev_visited_paths)
			self.visited_paths = visited_paths 
			self.src_cui_emb = new_src_cuis_emb 

			del visited_path_embs , paths_srcs, paths_tgt_edges 

		return final_scores, visited_paths, candidate_paths_df, stop_flag


# ====================== trainer  ===================
class Trainer(nn.Module):
	def __init__(self, tokenizer, 
				encoder, 
				g, 
				 vocab_emb_file, 
				 hdim, 
				 nums_of_head, 
				 all_edge_mappings,  
				 cui_aui_mappings, 
				 cui_vocab,
				 nums_of_hops, 
				 top_n,
				 device, 
				 nums_of_epochs,
				 LR, 
				 contrastive_learning=True,
				 save_model_path=None,
				 save_cui_embedding_path=None,
				 gnn_update=True, 
				 cui_flag=True,
				 intermediate=False,
				 distance_metric="Cosine",
				 path_encoder_type="Transformer",
				 gnn_type="Stack",
				 prune_thsh=0.8):
		super(Trainer, self).__init__() 

		self.tokenizer = tokenizer 
		self.encoder = encoder
		self.CUI_encoder = CuiEmbedding(vocab_emb_file) # store  
		self.gmodel = GraphModel(g, 
								 self.CUI_encoder,
								 hdim, 
								 nums_of_head, 
								 all_edge_mappings,
								 cui_aui_mappings, 
								 nums_of_hops,
								 top_n,
								 device, 
								gnn_update=gnn_update, 
								cui_flag=cui_flag,
								path_encoder_type=path_encoder_type,
								prune_thsh=prune_thsh,
								gnn_type="Stack") 
		# sync_networks(self.encoder) # sync encoder
		# sync_networks(self.gmodel)  # sync gmodel
		self.device = device 
		self.LR = LR 
		self.adam_epsilon =  0.99
		self.weight_decay =  0.99
		self.adam_epsilon = 1e-8
		self.nums_of_epochs = nums_of_epochs
		self.vocab_emb_file = vocab_emb_file 
		self.intermediate = intermediate # if computing loss on intermediate loss 
		self.print_step = 128 
		self.distance_metric = distance_metric
		self.prune_thsh = prune_thsh # prune threshold
		self.mode = 'train' 

		self.g = g 
		self.loss_fn = nn.BCEWithLogitsLoss() 
		self.cui_vocab = cui_vocab
		self.rev_cui_vocab = {v:k for k, v in self.cui_vocab.items()}
		self.batch_loss = torch.tensor(0).float().to(self.device) 
		self.k = nums_of_hops
		self.save_cui_embedding_path = save_cui_embedding_path
		self.save_model_path=save_model_path
		self.contrastive_learning = contrastive_learning 

		print("**** ============= **** ")

		exp_setting = f"TRAINER SETUP: SAVE CUI EMBEDDING {save_cui_embedding_path} \n NUMS OF HOPS: {nums_of_hops} \n TOP N NODES PER HOP: {top_n} \n COMPUTE INTERMEDIATE LOSS: {self.intermediate} \n LEARNING RATE: {LR} \n GNN UPDATE: {gnn_update} \n GNN TYPE: {gnn_type} \n CUI FLAG: {cui_flag} \n CONTRASTIVE LEARNING {contrastive_learning} \n PATH ENCODER TYPE: {path_encoder_type} TRIPLET LOSS DISTANCE METRIC:{self.distance_metric}"
		logging.info(exp_setting)
		print(exp_setting)

		print("**** ============= **** ")


	def create_optimizers(self):
		no_decay = ["bias", "LayerNorm.weight"]
		#self.model_params = list(self.gmodel.parameters()) 
		self.model_params = list(self.gmodel.parameters()) + [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)]
		print(f"LR: {self.LR}")
		if type(self.LR) == float:
			self.optimizer = torch.optim.AdamW(self.model_params, lr=self.LR,weight_decay=1e-4)
		else:
			self.optimizer = torch.optim.AdamW(self.model_params, lr=self.LR[0],weight_decay=1e-4)
		#self. # change to use scheduler later 

	def compute_triplet_loss(self, task_emb, start_cui_idx, cand_df_tgt, gold_idx, margin=1.0, distance_metric="Cosine"):
		"""
		Input: task_emb, 
		src_df_dicts: the data structure from gmodel gnn 
		start_cui_idx: initial starting nodes 
		cand_df_tgt: supposedly this should be candidate_df['Tgt'].tolist()
		gold_idx: ground truth idx of CUI in Vocab 
		anchor embedding: mean over task emb + src emb, where src emb is the updated GIN representation
		postive embedding: tgt embedding where tgt idx == gold idx
		negative embedding: tgt embedding where tgt idx != gold idx

		Output: triplet loss
		"""
		src_embedding = torch.mean(self.gmodel.n_encoder.encode(start_cui_idx), dim=0).unsqueeze(0).to(self.device) # shape b x hdim 
		anchor_embedding = task_emb * src_embedding # hadmard product between task and starting nodes
		for k,v in self.gmodel.gnn.src_df_dicts.items(): # k: cui idx; v: idx of the target nodes in the df    
			tgt_cui_labels = [cand_df_tgt[_] for _ in v]
			tgt_pos_idx = [self.cui_vocab[tgt_cui] for tgt_cui in tgt_cui_labels if self.cui_vocab[tgt_cui] in gold_idx]
			tgt_neg_idx = [self.cui_vocab[tgt_cui] for tgt_cui in tgt_cui_labels if self.cui_vocab[tgt_cui] not in gold_idx]

			if len(tgt_neg_idx) > 128: # set it as 64 to save memory
				sample_tgt_neg_idx = random.sample(tgt_neg_idx, 128) # only compute 100 negative samples
			else:
				sample_tgt_neg_idx = tgt_neg_idx

			pos_distance, neg_distance = 0, 0
			for _, t in enumerate(tgt_pos_idx):
				pos_emb = self.gmodel.n_encoder.encode([tgt_cui_labels[_]]).to(self.device)
				if distance_metric == "Cosine":
					pos_distance += torch.sum(nn.CosineSimilarity(dim=1, eps=1e-6)(anchor_embedding, pos_emb), dim=-1) 
				else:
					pos_distance += torch.sum(F.pairwise_distance(anchor_embedding, pos_emb, p=2), dim=-1) # Euclidean distance
			neg_embs = [] 
			for  _, t in enumerate(sample_tgt_neg_idx):
				neg_embs.append(self.gmodel.n_encoder.encode([tgt_cui_labels[_]]))

			if len(neg_embs) == 0:
				continue 
			else: 
				neg_embs = torch.mean(torch.stack(neg_embs).squeeze(1), dim=0).unsqueeze(0).to(self.device) # taking mean over sampled negative embedding
				if distance_metric == "Cosine":
					pos_distance += torch.sum(nn.CosineSimilarity(dim=1, eps=1e-6)(anchor_embedding, neg_embs), dim=-1) 
				else: 
					neg_distance = torch.sum(F.pairwise_distance(anchor_embedding, neg_embs, p=2), dim=-1) 
				loss = torch.max(pos_distance-neg_distance+margin, torch.tensor(0.0).to(self.device))
			self.batch_loss += loss 

		del anchor_embedding, neg_embs 

	def proj_to_vocab_compute_loss(self, entire_vocab, cui_idx_tensor, gold_idx_tensor): 
		"""
		Loss compute based on BCE: Loss = sum_i sum_j -(y_i,j * log(p_i,j) + (1 - y_i,j) * log(1 - p_i,j)) 
		"""

		for k, ele in enumerate(cui_idx_tensor):
			if ele in gold_idx_tensor[0]:
				loss = self.loss_fn(entire_vocab[:, ele], torch.tensor([1.]).to(self.device))
			else:
				loss = self.loss_fn(entire_vocab[:, ele], torch.tensor([0.]).to(self.device))
			self.batch_loss += loss

	def forward_per_round(self, task_emb, context_emb, cui_lists, labels_idx_per_sample, k2_idx_per_sample, k3_idx_per_sample):
		""" Perform step inference here; 
		Changes made Jan 13: add hard self contraint for self loop 
		"""
		running_k = 0 
		return_cui_lists = [] 
		init_cui_lists = cui_lists # the starting cui lists 
		#print(f"Specified number of hops: {self.k}")
		self.gmodel.candidate_paths_df = None 
		prev_visited_paths = {}
		prev_final_scores = None 
		prev_cand_cuis = [] 
		prev_cand_path_df = None 
		entire_vocab = torch.zeros(1, len(cui_vocab)).to(self.device) 

		for running_k in range(self.k):
			final_scores, visited_paths, candidate_paths_df, stop_flag = self.gmodel.one_iteration(task_emb, 
																								   cui_lists, 
																								   running_k, 
																								   context_emb=context_emb,
																								   prune_thsh=self.prune_thsh)  
			cui_lists = list(self.gmodel.src_cui_emb.keys())

			if not stop_flag:
				prev_visited_paths.update(visited_paths) # dictionary update 
				prev_final_scores = final_scores 
				prev_cand_path_df = candidate_paths_df
				#print("Visited Paths", prev_visited_paths)

				if running_k == self.k-1: 
					return_cui_lists = cui_lists

				cand_cuis = candidate_paths_df['Tgt'].tolist() # final prediction; CUI IDs 

			cui_idx = [cui_vocab[k] for k in cand_cuis] 
			cui_idx_tensor = torch.tensor(cui_idx)
			entire_vocab = torch.zeros(1, len(cui_vocab)).to(self.device)
			for k, ele in enumerate(cui_idx_tensor):
				entire_vocab[:,ele] += prev_final_scores[k]

			tgtlabels = torch.tensor([labels_idx_per_sample]) # this is k2_idx[_], a list of final nodes

			## If intermediate path loss is True and we are using intermediate ground truth labels for loss 
			if self.intermediate == True:
				if running_k == 0:
					tgtlabels = torch.tensor([k2_idx_per_sample])
				elif running_k == 1 and self.k == 3:
					 tgtlabels = torch.tensor([k2_idx_per_sample])
			 
			self.proj_to_vocab_compute_loss(entire_vocab,cui_idx_tensor, tgtlabels) 
			if self.contrastive_learning and self.mode == "train": 
				self.compute_triplet_loss(task_emb, init_cui_lists, cand_cuis, tgtlabels)

		return prev_visited_paths, prev_cand_cuis, prev_cand_path_df

	def measure_accuracy(self, gold_idx, batch_visited_paths, mode="Recall"):
		B = len(gold_idx)
		accs = []
		for _ in range(B):
			gold_cuis = [self.rev_cui_vocab[i] for i in gold_idx[_]]
			pred_cuis = [i for i in batch_visited_paths[_]]
			if mode == "Precision":
				# Precision hit rate 
				acc = len(set(gold_cuis).intersection(set(pred_cuis))) / len(set(pred_cuis))
			elif mode == "Recall":
				# Recall hit rate 
				acc = len(set(gold_cuis).intersection(set(pred_cuis))) / len(set(gold_cuis))
			else:
				prec = len(set(gold_cuis).intersection(set(pred_cuis))) / len(set(pred_cuis)) 
				rec = len(set(gold_cuis).intersection(set(pred_cuis))) / len(set(gold_cuis)) 
				acc = 2*(prec*rec) / (prec+rec)
			accs.append(acc)
		return np.mean(accs) 


	def forward_per_batch(self, batch):

		rst_cui_tks_ids, rst_cui_tks_attn, rst_text_tks_ids, rst_text_tks_attn, input_cuis_idx, labels_idx, k2_idx, k3_idx  = batch 
		input_task_embs = self.encoder(rst_text_tks_ids.to(self.device).squeeze(0), 
								   rst_text_tks_attn.to(self.device).squeeze(0)).pooler_output
		input_cui_embs = self.encoder(rst_cui_tks_ids.to(self.device).squeeze(0), 
								  rst_cui_tks_attn.to(self.device).squeeze(0)).pooler_output

		B = input_cui_embs.shape[0]
		self.batch_loss = torch.tensor(0.0).to(self.device)
		batch_visited_paths = [] 

		for _ in range(B):
			input_cui_lists = [self.rev_cui_vocab[i] for i in input_cuis_idx[_]]
			task_embs = input_task_embs[_,:].unsqueeze(0)
			cui_embs = input_cui_embs[_, :].unsqueeze(0)

			visited_paths, cand_cuis, candidate_paths_df = self.forward_per_round(task_embs, 
																				cui_embs, 
																				  input_cui_lists, 
																				  labels_idx[_], 
																				 k2_idx[_],
																				  k3_idx[_],
																				 )
			batch_visited_paths.append(visited_paths) 

		accs = self.measure_accuracy(labels_idx, batch_visited_paths)
		del input_task_embs, input_cui_embs, task_embs,  cui_embs
		return self.batch_loss.item(), batch_visited_paths, accs

	def validate(self, dev_data):
		with torch.no_grad():
			ep_loss_dev = [] 
			ep_acc_dev = [] 
			cnt = 0 
			dev_label_idx = [] 
			visited_paths_dev = [] 
			for batch in tqdm(dev_data):
				cnt +=1 
				if cnt > 300:
					continue
				rst_cui_tks_ids, rst_cui_tks_attn, rst_text_tks_ids, rst_text_tks_attn, input_cuis_idx, labels_idx, k2_idx, k3_idx  = batch 

				input_task_embs = self.encoder(rst_text_tks_ids.to(self.device).squeeze(0), 
											   rst_text_tks_attn.to(self.device).squeeze(0)).pooler_output
				input_cui_embs = self.encoder(rst_cui_tks_ids.to(self.device).squeeze(0), 
											  rst_cui_tks_attn.to(self.device).squeeze(0)).pooler_output

				B = input_cui_embs.shape[0]
				self.batch_loss = torch.tensor(0.0).to(self.device)
				batch_visited_paths = [] 

				for _ in range(B):
					input_cui_lists = [self.rev_cui_vocab[i] for i in input_cuis_idx[_]]
					task_embs = input_task_embs[_,:].unsqueeze(0)
					cui_embs = input_cui_embs[_, :].unsqueeze(0)

					visited_paths, cand_cuis, candidate_paths_df = self.forward_per_round(task_embs, 
																						cui_embs, 
																						  input_cui_lists, 
																						  labels_idx[_], 
																						 k2_idx[_],
																						  k3_idx[_]
																						 )
					batch_visited_paths.append(visited_paths) 


				accs = self.measure_accuracy(labels_idx, batch_visited_paths)
				dev_label_idx.append(labels_idx)
				visited_paths_dev.append(batch_visited_paths)
				batch_loss = self.batch_loss.item()
				ep_loss_dev.append(batch_loss)
				ep_acc_dev.append(accs)
			this_ep_loss_dev = np.mean(ep_loss_dev)
			this_ep_acc_dev = np.mean(ep_acc_dev)
			del input_task_embs, input_cui_embs
		return this_ep_loss_dev, this_ep_acc_dev, dev_label_idx, visited_paths_dev

	def train(self, train_data, dev_data, lr_scheduler): 
		# Input: train dataloader, dev_dataloader 

		if not self.optimizer:
			self.create_optimizers()

		all_epoch_loss = [] 
		min_epoch_loss = 1000 
		min_epoch_loss_train = min_epoch_loss 
		update_step = 4
		evaluate_step = int(len(train_data) / 4) # save two evaluation per epoch  

		for ep in range(self.nums_of_epochs): 
			lr_scheduler.step() 
			#self.gmodel.train()
			ep_loss_train = []
			cnt = 0
			ep_acc_train = [] 
			for batch in tqdm(train_data):
				batch_loss, batch_visited_paths, accs = self.forward_per_batch(batch)
				self.batch_loss = self.batch_loss / update_step
				self.batch_loss.backward()
				# sync_grads(self.model_params) # sync gradients across processes
				cnt +=1  
				if cnt % update_step == 0:
					torch.nn.utils.clip_grad_norm_(self.model_params, 3.0) 

					self.optimizer.step()
					self.optimizer.zero_grad()
					self.batch_loss = 0 

				ep_loss_train.append(batch_loss)
				ep_acc_train.append(accs)

				if cnt % self.print_step == 0:
					step_loss = np.mean(ep_loss_train)
					step_acc = np.mean(ep_acc_train)
					print_statement = f"EPOCH {ep} Training Step {cnt} Loss: {step_loss:.4f} Acc: {step_acc:.4f}" 
					print(print_statement)
					logging.info(print_statement) 

				if cnt % evaluate_step == 0 :
					this_ep_loss_dev, this_ep_acc_dev, dev_label_idx, visited_paths_dev= self.validate(dev_data)
					print_statement = f"STEP {cnt} VAL Loss: {this_ep_loss_dev:.4f} VAL Acc: {this_ep_acc_dev:.4f}" 
					print(print_statement)
					logging.info(print_statement)
					if this_ep_loss_dev < min_epoch_loss:
						min_epoch_loss = this_ep_loss_dev
						if self.save_model_path:
							# save network components
							torch.save(self.gmodel.state_dict(), self.save_model_path)
							#torch.save(self.gmodel.state_dict(), self.save_model_path)
							torch.save(self.encoder.state_dict(), self.save_model_path[:self.save_model_path.rfind("/")]+"/encoder.pth")
							# save node embedding layer
							with open(self.save_cui_embedding_path,'wb') as outf:
								pickle.dump(self.gmodel.n_encoder.data, outf) 

							with open(self.save_model_path[:self.save_model_path.rfind("/")]+"/dev_output.json","w") as outf:
								json.dump([dev_label_idx, visited_paths_dev], outf)

			#lr_scheduler.step()
			# if MPI.COMM_WORLD.Get_rank() == 0: # print training loss and accuracy based on rank 0
			print('epoch={}, learning rate={:.5f}'.format(ep, self.optimizer.state_dict()['param_groups'][0]['lr']))
			logging.info('epoch={}, learning rate={:.5f}'.format(ep, self.optimizer.state_dict()['param_groups'][0]['lr']))

			this_ep_loss_train = np.mean(ep_loss_train)
			this_ep_loss_dev, this_ep_acc_dev, dev_label_idx, visited_paths_dev= self.validate(dev_data)

			if this_ep_loss_dev < min_epoch_loss:
				min_epoch_loss = this_ep_loss_dev
				# save network components
				torch.save(self.gmodel.state_dict(), self.save_model_path)
				# save node embedding layer
				with open(self.save_cui_embedding_path,'wb') as outf:
					pickle.dump(self.gmodel.n_encoder.data, outf) 
					torch.save(self.encoder.state_dict(), self.save_model_path[:self.save_model_path.rfind("/")]+"/encoder.pth")

				with open(self.save_model_path[:self.save_model_path.rfind("/")]+"/dev_output.json","w") as outf:
					json.dump([dev_label_idx, visited_paths_dev], outf)

			all_epoch_loss.append(this_ep_loss_dev)

			print(f"Epoch {ep} Training loss {this_ep_loss_train:.4f} Val Loss {this_ep_loss_dev:.4f} Val Acc {this_ep_acc_dev*100:.4f} ")
			logging.info(f"Epoch {ep} Training loss {this_ep_loss_train:.4f} Val Loss {this_ep_loss_dev:.4f} Val Acc {this_ep_acc_dev*100:.4f}") 



class TestSet(Dataset):
	""" files in json format 
	"""
	def __init__(self, datapath, tokenizer, cui_vocab, cui_flag=True,intermediate=True,oracle=False,k=None):
		self.data = json.load(open(datapath,'r')) 
		self.all_keys = list(self.data.keys())
		self.tokenizer = tokenizer 
		self.cui_vocab = cui_vocab 
		self.intermediate = intermediate
		self.cui_flag = cui_flag
		self.k = k # 2 or 3 if not None 
		self.oracle = oracle 

	def __len__(self):
		return len(self.data)


	def __getitem__(self, index):
		k = self.all_keys[index]
		text = self.data[k]['input text'] # input text 
		# place holder for input context
		cuis_ids = [self.cui_vocab[i] for i in self.data[k]['input A CUI']] # input candidate cuis
		paths = self.data[k]['paths']
		if self.cui_flag and 'input context' in self.data[k]:
			concept = " [SEP] ".join(list(set(self.data[k]['input context'])))
		else:
			concept = text 
		intermediate_golds = [] 
		k2_golds = []
		k3_golds = [] 
		input_cui_tks = self.tokenizer(concept, 
									truncation=True, 
									padding="max_length", 
									max_length=256, 
									pad_to_max_length=True,
									return_tensors="pt")
		input_text_tks = self.tokenizer(text, 
										truncation=True, 
										padding="max_length",
										max_length=256, 
										pad_to_max_length=True,
										return_tensors="pt")

		golds = [] 
		golds = [self.cui_vocab[p[-1]] for p in paths if self.cui_vocab[p[-1]] not in golds]
		if self.oracle: # if in a oracel exp setting 
			starting_nodes = [] 
			starting_nodes = [self.cui_vocab[p[0]] for p in paths if self.cui_vocab[p[0]] not in starting_nodes]
			cuis_ids = starting_nodes 

		return input_cui_tks, input_text_tks, cuis_ids, golds, k2_golds, k3_golds 


nums_of_hops = 2
top_n = 16
gnn_update=True
intermediate = False
batch_size = 1 
cui_flag=True
path_encoder_type = "MLP" 
oracle = False # if it is an oracle experiment

pretrain_vocab_embedding = "GraphModel_SNOMED_CUI_Embedding.pkl" 
distance_metric="Cosine"

pretrained_model_path = "MIMIC_Models/k2_N8_H3_CL_V4_CosinewBERT_stack_Oracle28-03-23-16/"

cui_aui_mappings = pickle.load(open("sm_t047_cui_aui_eng.pkl","rb")) 
g = pickle.load(open("SNOMED_CUI_MAJID_Graph_wSelf.pkl","rb"))
edge_labels=[]
for edges in list(g.edges()):
	edge_label = g.get_edge_data(*edges)['label']
	edge_labels.append(edge_label)

all_edge_mappings = {k: v for v, k in enumerate(sorted(set(edge_labels)))}
len(all_edge_mappings)
node_labels = list(g.nodes())

cui_vocab = {k: v for v, k in enumerate(set(node_labels))} 
trainer = Trainer(tokenizer, 
				  model, 
				  g, 
				  pretrain_vocab_embedding, 
				  768, 
				  3, 
				  all_edge_mappings, 
				  cui_aui_mappings, 
				  cui_vocab, 
				  nums_of_hops, 
				  top_n, 
				  torch.device('cuda'), 
				  1,
				  1e-4, 
				  intermediate=intermediate,
				  cui_flag=cui_flag, 
				  gnn_update=True, 
				  path_encoder_type=path_encoder_type,
				  distance_metric=distance_metric,
				  prune_thsh=0.8)

trainer.to(torch.device('cuda'))

trainer.gmodel.load_state_dict(torch.load(pretrained_model_path+"gmodel.pth"))
trainer.encoder.load_state_dict(torch.load(pretrained_model_path+"encoder.pth"))

#trainer.gmodel.eval()
#trainer.encoder.eval() 

print(f"Loading previous checkpoints from {pretrained_model_path}")

dataset = TestSet("Summ_test.json", tokenizer, cui_vocab, intermediate=False, k=2, oracle=oracle)
loader=DataLoader(dataset, collate_fn=collate_fn, batch_size=1, shuffle=False) 

trainer.mode = "test"
dev_loss, dev_acc, dev_label_idx, visited_paths_dev = trainer.validate(loader)
print("Acc: ", dev_acc) 
with open(pretrained_model_path+"Summ_test_output"+str(oracle)+"_K_"+str(nums_of_hops)+"_N_"+str(top_n)+"_stackgnn.json","w") as outf:
	json.dump(visited_paths_dev, outf) 