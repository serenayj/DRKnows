# DR.KNOWs: Diagnostic Reasoning Knowledge Graphs for LLM Diagnosis Prediction 

This is the code repository for our DR.Knows Paper. Please cite us if you are using this repo: 

Gao Y, Li R, Caskey J, Dligach D, Miller T, Churpek MM, Afshar M. Leveraging a medical knowledge graph into large language models for diagnosis prediction. arXiv preprint arXiv:2308.14321. 2023 Aug 28. 

OR 

Gao Y, Li R, Croxford E, Tesch S, To D, Caskey J, W. Patterson B, M. Churpek M, Miller T, Dligach D, Afshar M. Large language models and medical knowledge grounding for diagnosis prediction. medRxiv. 2023 Nov 27:2023-11.

The paper is currently under reviewed. 

## Step 0: Setup QuickUMLS 

## Step 1: Download pre-build KG and CUI embedding

Pre-build UMLS SNOMED CUI graph object (with physician selected relations pertinent to diagnosis): [download](https://drive.google.com/file/d/1zlb0zey_tAnFWtCY_NvhA0dqfydL4Ph7/view?usp=sharing) This file is about 700 MB. 

Pre-build Graph CUI Embedding (generated from SapBERT encoder): [download](https://drive.google.com/file/d/1a2axTk35wsvQ4AJOheZnjZJdyksHg1cy/view?usp=sharing) For each CUI, we generated embedding using `CUI Preferred Text 1 [SEP] CUI Preferred Text 2 [SEP] ... [SEP]`. This gives us the best semantic representation in our preliminary experiments. This file is about 1.3 GB. 

You could also choose to generate your own CUI embedding file, using the `generate_CUI_embedding.py` script. 

## Step 2: Training

We offer two trainer for DR.Knows: Multi-head Attention and Trilinear Attention. Both trainer scripts are optimized for MPI (training reduced to 2-3 hours on 5k notes input), so please have [mpi4py](https://mpi4py.readthedocs.io/en/stable/) installed. 

## Step 3: Inference 


