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

## Dataset

We use HuggingFace datasets library to access and save each dataset.
We save it as .txt file for the sources, and another one for the summaries, with 1 data point per line.
For CNN/DM, we save one .txt file for every single data point.

For instance to download and save Reddit:
```
cd src/candidate_generation/
bash dataset.sh
```

Note that for Reddit, we make a custom 80/10/10 train/val/test split.

## EVALUATION pipeline (assumes an already trained SummaReranker checkpoint)

### 1 - Generate summary candidates
SummaReranker takes as input a set of summary candidates from a given sequence-to-sequence model (PEGASUS, BART) and a given decoding method
(beam search, diverse beam search, top-p sampling, top-k sampling).  

For instance with PEGASUS on Reddit validation set, and with diverse beam search:
```
CUDA_VISIBLE_DEVICES=0 bash candidate_generation.sh
```
Generating summary candidates should take a few hours on the validation or test sets of CNN/DM, XSum or Reddit.

Note that for Reddit, you need to fine-tune the model on your training split prior to generating candidates.

### 2 - Score the candidates
To evaluate SummaReranker, we need to score each summary candidate with all the metrics of interest (ROUGE-1/2/L, BERTScore, BARTScore, etc).  

For instance to score PEGASUS diverse beam search candidates on Reddit validation set with ROUGE-1:
```
CUDA_VISIBLE_DEVICES=0 bash scores.sh
```
Scoring all candidates should take a few minutes with ROUGE metrics on the validation or test sets of CNN/DM, XSum or Reddit. 

### 3 - Download the model checkpoint
CNN/DM checkpoint (trained on beam search + diverse beam search candidates, for ROUGE-1/2/L metrics): <a href="https://drive.google.com/file/d/1aHX6Piehyp2hV59le-ccsmR56pUbOttx/view?usp=sharing" style = "text-decoration:none;color:#4682B4">here</a>  
XSum checkpoint (trained on beam search + diverse beam search candidates, for ROUGE-1/2/L metrics): <a href="https://drive.google.com/file/d/1bwlpqFixw1iLXrOxIOB3GjuQSNeHG9_6/view?usp=sharing" style = "text-decoration:none;color:#4682B4">here</a>   
Reddit checkpoint (trained on beam search + diverse beam search candidates, for ROUGE-1/2/L metrics): <a href="https://drive.google.com/file/d/11aXfXtVNGOpawNUHBqSp-gaot9-NexGG/view?usp=sharing" style = "text-decoration:none;color:#4682B4">here</a>  

### 4 - Run SummaReranker
For instance, to run SummaReranker trained for ROUGE-1/2/L on PEGASUS (beam search + diverse beam search candidates) on Reddit validation set:
```
cd ../summareranker/
CUDA_VISIBLE_DEVICES=0 bash evaluate.sh
```
Make sure that the argument --load_model_path points to where you placed the SummaReranker checkpoint. 

## TRAINING pipeline

### 1 - Fine-tune base models

For training, SummaReranker follows a cross-validation approach: the training set is split in two, and we train one model on each half, to then infer it and use its predictions on the other half. We also need a third model trained on the entire training set (for the transfer setup at inference time), which we re-train ourselves for Reddit. 

For instance with PEGASUS on Reddit:
```
cd ../base_model_finetuning/
CUDA_VISIBLE_DEVICES=0 bash train_base_models.sh
```
Note that this single script performs all the tasks of splitting the training set, then training models and on each half, and training a model on the entire set.

### 2 - Generate summary candidates
Then, we need to get summary candidates on the training, validation and test sets. 

For instance with PEGASUS on Reddit with diverse beam search:
```
cd ../candidate_generation/
CUDA_VISIBLE_DEVICES=0 bash candidate_generation_train.sh
```
Generating summary candidates on the entire datasets should take *up to a few days*.

### 3 - Score the candidates
Next, we need to score the summary candidates on the training, validation and test sets for each of the metrics.

For instance to score PEGASUS diverse beam search candidates on Reddit with ROUGE-1/2/L:
```
CUDA_VISIBLE_DEVICES=0 bash scores_train.sh
```
Scoring all candidates should take a few minutes with ROUGE metrics. 

### 4 - Train SummaReranker
For instance, to train SummaReranker trained for ROUGE-1/2/L on PEGASUS diverse beam search candidates on Reddit:
```
cd ../summareranker/
CUDA_VISIBLE_DEVICES=0 bash train.sh
```

## Citation
If you find our paper or this project helps your research, please kindly consider citing our paper in your publication.   
```
@article{ravaut2022summareranker,
  title={SummaReranker: A Multi-Task Mixture-of-Experts Re-ranking Framework for Abstractive Summarization},
  author={Ravaut, Mathieu and Joty, Shafiq and Chen, Nancy F},
  journal={arXiv preprint arXiv:2203.06569},
  year={2022}
}
```
