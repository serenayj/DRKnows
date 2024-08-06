from quickumls import *
import pickle
import networkx as nx
import json 
from tqdm import tqdm 
import sys 
import pandas as pd 

batch_idx = sys.argv[1] 

cui_aui_mappings = pickle.load(open("sm_t047_cui_aui_eng.pkl","rb")) 
g = pickle.load(open('SNOMED_CUI_MAJID_Graph_wSelf.gpickle','rb'))
matcher = QuickUMLS("/Users/ygao/Downloads/2021AB/yanjun",overlapping_criteria="score",similarity_name="jaccard", threshold=0.9) 
# Graph utils functions 
#progress_notes_new = pd.read_csv("MIMIC_Progress_Notes.csv", header=None)
#progress_notes = progress_notes_new.values.tolist()

progress_notes = json.load(open("All_Progress_Notes.json", "r"))

def retrieve_cuis(text,g, matcher):
	# Retrieve cuis from quickUMLS 
	output = matcher.match(text)
	#output
	cui_output= [ii['cui'] for i in output for ii in i if ii['cui'] in g.nodes]
	terms = [ii['term'] for i in output for ii in i if ii['cui'] in g.nodes]
	cui_outputs = set(cui_output)
	
	# answer: C0010346 
	return cui_outputs, output, terms

def retrieve_cuis_semantictype(text,g, matcher, semantic_type):
	# Retrieve cuis from quickUMLS 
	output = matcher.match(text)
	#output
	#cui_output= [ii['cui'] for i in output for ii in i if ii['cui'] in g.nodes if semantic_type in list(ii['semtypes'])]
	#terms = [ii['term'] for i in output for ii in i if ii['cui'] in g.nodes if semantic_type in list(ii['semtypes'])]
	cui_outputs = []
	terms = []
	for i in output :
		for ii in i:
			if ii['cui'] in g.nodes and semantic_type in list(ii['semtypes']):
				cui_outputs.append(ii['cui'])
				terms.append(ii['term'])
	cui_outputs = set(cui_outputs)
	
	# answer: C0010346 
	return cui_outputs, output, terms


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

def retrieve_neighbors_paths(cui_lists, g):
	cui_neighbors = retrieve_subgraphs(cui_lists, g) # dictionary of cuis and their neighrbos 
	all_neighbors = [] 
	all_paths = [] 
	for k,v in cui_neighbors.items():
		if len(v) != 0 :
			for vv in v:
				all_neighbors.append(vv[0]) # list of neighbor nodes 
				all_paths.append([k, vv[0], vv[1]]) # list of one-hop path 
	return all_paths, all_neighbors

# pipeline for pre-processing distant supervision data


def SelectPaths(cui_path, cui_aui_mappings, g, max_k=4):
	outputs = [] 
	for p in cui_path:
		#strs = [cui_aui_mappings[pp][0][1] for pp in p]
		#print("PATH: {}".format(strs))
		if len(p) > max_k: 
			continue 
		#strs = f"PATH with {max_k-1} Hops: " + cui_aui_mappings[p[0]][0][1]
		p_tmp = [p[0]]
		for s in range(len(p)):
			if s < len(p)-1:
				edge = g.get_edge_data(p[s], p[s+1])['label']
				p_tmp.extend([edge, p[s+1]]) 
				#strs +=  "\t"*s +" -- "+ edge + " --> "+ cui_aui_mappings[p[s+1]][0][1] + "\n"
		#print(strs)
		outputs.append(p_tmp)
	return outputs 

def build_cui_paths(a_cuis, all_p_cuis, g):
	# added a check for shortest path length
	all_paths = [] 
	for a_c in list(set(a_cuis)):
		for p_c in all_p_cuis:
			min_l = 0 
			if nx.has_path(g, a_c, p_c):
				if nx.shortest_path_length(g, source=a_c, target=p_c) > 2:
					continue 
				paths = list(nx.all_shortest_paths(g, a_c, p_c))
				min_l = min([len(p) for p in paths])
				#all_paths.append(paths)
			elif nx.has_path(g, p_c, a_c):
				if nx.shortest_path_length(g, source=p_c, target=a_c) > 2:
					continue 
				paths = list(nx.all_shortest_paths(g, p_c, a_c))
				min_l = min([len(p) for p in paths])
				#all_paths.append(paths)
			if min_l != 0:
				short_paths = [p for p in paths if len(p) == min_l]
				all_paths.extend(short_paths)
	return all_paths


if __name__ == "__main__":
	batch_begins = int(batch_idx) * 1500 
	batch_ends = batch_begins + 1500 
	print(f"Processing notes {batch_begins} to {batch_ends}")
	all_output_paths = {}
	for _, record in enumerate(tqdm(progress_notes[batch_begins:batch_ends])): 
		note = record[10]
		row_id = record[0]
		_ = row_id
		if "Assessment and Plan" in note: 
			#list([i.split("Assessment and Plan")[-1] for i in increasing['TEXT']])
			aps = note.split("Assessment and Plan")[-1] 
			a = aps.split("#")[0]
			ps= " ".join(aps.split("#")[1:])
			a_cui_outputs, a_output, a_terms = retrieve_cuis(a, g, matcher) 
			eg_p_cuis, eg_p_outputs, eg_p_terms = retrieve_cuis_semantictype(ps, g, matcher, "T047") 
			#if len(eg_p_)
			eg_p_cuis = list(set(eg_p_cuis))
			eg_a_cuis = list(set(a_cui_outputs))
			if len(eg_p_cuis) > 10: # skip the examples where number of cuis in P over 10
				continue 
			if len(eg_p_cuis) > 0 and len(eg_a_cuis) > 0:
				all_paths = build_cui_paths(eg_a_cuis, eg_p_cuis, g) 
				#print(len(all_paths)) 
				if len(all_paths) > 0 :
					outputs= SelectPaths(all_paths, cui_aui_mappings, g, 3) 
					if len(outputs) > 0:
						all_output_paths[_] = {}
						all_output_paths[_]['input text'] = a 
						all_output_paths[_]['input A CUI'] = list(set(eg_a_cuis)) 
						all_output_paths[_]['input P CUI'] = list(set(eg_a_cuis))
						all_output_paths[_]['input context'] = a_terms 
						all_output_paths[_]['paths'] = outputs
				else:
					continue 

	with open(f"MIMIC_Paths/batch_{str(batch_begins)}.json", "w") as jout:
		json.dump(all_output_paths, jout) 

					
