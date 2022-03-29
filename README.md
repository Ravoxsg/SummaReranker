# SummaReranker
Source code for the paper <a href="https://arxiv.org/pdf/2203.06569.pdf" style = "text-decoration:none;color:#4682B4">SummaReranker: A Multi-Task Mixture-of-Experts Re-ranking Framework for Abstractive Summarization</a>.

Mathieu Ravaut, Shafiq Joty, Nancy F. Chen.

Accepted for publication at ACL 2022. 

## Setup

### 1 - Download the code
```
git clone https://github.com/Ravoxsg/SummaReranker.git
cd SummaReranker
```

### 2 - Install the dependencies
```
conda create --name summa_reranker python=3.8.8
conda activate summa_reranker
pip install -r requirements.txt
```

## Generate the re-ranking dataset

### 1 - Generate summary candidates
SummaReranker takes as input a set of summary candidates from a given sequence-to-sequence model (PEGASUS, BART) and a given decoding method
(beam search, diverse beam search, top-p sampling, top-k sampling).  
For instance with PEGASUS on Reddit validation set, and with diverse beam search:
```
cd src/candidate_generation/
CUDA_VISIBLE_DEVICES=0 bash candidate_generation.sh
```
Generating summary candidates should take a few hours on the test set of CNN/DM, XSum or Reddit.

Note that for Reddit, you need to fine-tune the model on your training split prior to generating candidates.

### 2 - Score the candidates
To evaluate SummaReranker, we need to score each summary candidate with all the metrics of interest (ROUGE, BERTScore, BARTScore, etc).  
For instance to score PEGASUS diverse beam search candidates on Reddit validation set with ROUGE-L:
```
CUDA_VISIBLE_DEVICES=0 bash scores.sh
```
Scoring all candidates should take a few minutes with ROUGE metrics on the test set of CNN/DM, XSum or Reddit. 

## Run SummaReranker

### 1 - Download the model checkpoint
CNN/DM checkpoint (trained on beam search + diverse beam search candidates, for ROUGE-1/2/L metrics): <a href="https://drive.google.com/file/d/1aHX6Piehyp2hV59le-ccsmR56pUbOttx/view?usp=sharing" style = "text-decoration:none;color:#4682B4">here</a>  
XSum checkpoint (trained on beam search + diverse beam search candidates, for ROUGE-1/2/L metrics):  
Reddit checkpoint (trained on beam search + diverse beam search candidates, for ROUGE-1/2/L metrics):  

### 2 - Run SummaReranker
For instance, to run SummaReranker trained for ROUGE-1/2/L on PEGASUS beam search candidates on 300 validation samples on Reddit:
```
cd ../summareranker/
CUDA_VISIBLE_DEVICES=0 bash evaluate.sh
```
Make sure that the argument --load_model_path points to where you placed the SummaReranker checkpoint. 
