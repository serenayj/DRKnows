# DR.KNOWs: Diagnostic Reasoning Knowledge Graphs for LLM Diagnosis Prediction 


## Step 0: Setup QuickUMLS 

## Step 1: Download pre-build KG and CUI embedding

Pre-build UMLS SNOMED CUI graph object (with physician selected relations pertinent to diagnosis): [download](https://drive.google.com/file/d/1zlb0zey_tAnFWtCY_NvhA0dqfydL4Ph7/view?usp=sharing)

Pre-build Graph CUI Embedding (generated from SapBERT encoder): [download]()

## Step 2: Training

We offer two trainer for DR.Knows: Multi-head Attention and Trilinear Attention. Both trainer scripts are optimized for MPI (training reduced to 2-3 hours on 5k notes input), so please have [mpi4py](https://mpi4py.readthedocs.io/en/stable/) installed. 

## Step 3: Inference 


