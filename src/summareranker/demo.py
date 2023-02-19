# Complete SummaReranker generation inference pipeline in a single small script!
# By default: CNN/DM, DBS with 15 candidates, SummaReranker trained for R-1/R-2/R-L optimization on BS + DBS candidates

import sys
import gc
sys.path.append("/data/mathieu/SummaReranker/src/")
import numpy as np
import torch
import argparse
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from transformers import PegasusTokenizer, PegasusTokenizerFast, PegasusForConditionalGeneration, RobertaTokenizerFast, RobertaModel
from rouge_score import rouge_scorer

from common.utils import seed_everything
from model import ModelMultitaskBinary


parser = argparse.ArgumentParser()
args = parser.parse_args()
args.device = torch.device("cuda")
args.generation_methods = ["diverse_beam_search"]
args.num_beams = 15
args.scoring_methods = ["rouge_1", "rouge_2", "rouge_l"]
args.filter_out_duplicates = True
args.sep_symbol = "[SEP]"
args.n_tasks = 3
args.hidden_size = 1024
args.use_shared_bottom = True
args.bottom_hidden_size = 1024
args.num_experts = 6
args.k = 3
args.expert_hidden_size = 1024
args.tower_hidden_size = 1024
args.sharp_pos = False
args.use_aux_loss = False
args.max_length = 512
args.max_source_length = 384
args.max_summary_length = 128

# seed
seed_everything(43)

# data
dataset_name = "ccdv/cnn_dailymail"
version = "3.0.0"
subset = "test"
text_key = "article"
summary_key = "highlights"
dataset = load_dataset(dataset_name, version, cache_dir="/data/mathieu/hf_datasets/")
dataset = dataset[subset]
texts = dataset[text_key]
labels = dataset[summary_key]
p = np.random.permutation(len(texts))
texts = [texts[x] for x in p]
text = texts[0]
text = text.replace("\n", " ")
print("\nSource document:")
print(text)
labels = [labels[x] for x in p]
label = labels[0]
label = "\n".join(sent_tokenize(label))
print("\nLabel:")
print(label)

# base model
base_model_name = "google/pegasus-cnn_dailymail"
base_tokenizer = PegasusTokenizer.from_pretrained(base_model_name, cache_dir="/data/mathieu/hf_models/pegasus-large-cnndm/")
base_model = PegasusForConditionalGeneration.from_pretrained(base_model_name, cache_dir="/data/mathieu/hf_models/pegasus-large-cnndm/")
base_model = base_model.to(args.device)
base_model = base_model.eval()

# candidates
tok_text = base_tokenizer(text, return_tensors="pt", padding="max_length", max_length=1024)
tok_text["input_ids"] = tok_text["input_ids"][:, :1024]
tok_text["attention_mask"] = tok_text["attention_mask"][:, :1024]
with torch.no_grad():
    generated = base_model.generate(
        input_ids=tok_text["input_ids"].to(args.device),
        attention_mask=tok_text["attention_mask"].to(args.device),
        num_beams=15,
        num_beam_groups=15,
        diversity_penalty=1.0,
        num_return_sequences=15,
        repetition_penalty=1.0,
        length_penalty=0.8,
        no_repeat_ngram_size=3
    )
candidates = base_tokenizer.batch_decode(generated, skip_special_tokens=True)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer = True)
print("\nSummary candidates:")
for j in range(len(candidates)):
    candidates[j] = candidates[j].replace("<n>", " ")
    candidates[j] = "\n".join(sent_tokenize(candidates[j]))
    rouge_scores = scorer.score(label, candidates[j])
    r1 = 100 * rouge_scores["rouge1"].fmeasure
    r2 = 100 * rouge_scores["rouge2"].fmeasure
    rl = 100 * rouge_scores["rougeLsum"].fmeasure
    mean_r = (r1 + r2 + rl) / 3
    print("\nCandidates {} (Mean R: {:.2f}, R-1: {:.2f}, R-2: {:.2f}, R-L: {:.2f})".format(j, mean_r, r1, r2, rl))
    candidates[j] = candidates[j].replace("\n", " ")
    print(candidates[j])

del base_model
del tok_text
del generated
gc.collect()

# SummaReranker
# model
model_name = "roberta-large"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name, cache_dir="/data/mathieu/hf_models/roberta-large/")
model = RobertaModel.from_pretrained(model_name, cache_dir="/data/mathieu/hf_models/roberta-large/")
model = model.to(args.device)
summareranker_model = ModelMultitaskBinary(model, tokenizer, args)
summareranker_model = summareranker_model.to(args.device)
summareranker_model_path = "/data/mathieu/2nd_stage_summarization/4_supervised_multitask_reranking/saved_models/cnndm/multitask_3_tasks_ablation_8/checkpoint-12500/pytorch_model.bin"
summareranker_model.load_state_dict(torch.load(summareranker_model_path))
summareranker_model = summareranker_model.eval()
# prepare the data
text_inputs = tokenizer(text, return_tensors="pt", max_length=args.max_source_length, padding='max_length')
text_inputs["input_ids"] = text_inputs["input_ids"][:, :args.max_source_length]
text_and_candidates_ids, text_and_candidates_masks = [], []
for j in range(len(candidates)):
    candidate = candidates[j]
    candidate_inputs = tokenizer(candidate, return_tensors="pt", max_length=args.max_summary_length, padding='max_length')
    candidate_inputs["input_ids"] = candidate_inputs["input_ids"][:, :args.max_summary_length]
    block = tokenizer.batch_decode(text_inputs["input_ids"], skip_special_tokens = True)[0] + args.sep_symbol + tokenizer.batch_decode(candidate_inputs["input_ids"], skip_special_tokens = True)[0]
    text_and_candidate = tokenizer(block, return_tensors="pt", padding="max_length", max_length=args.max_length)
    ids = text_and_candidate["input_ids"][:, :args.max_length]
    mask = text_and_candidate["attention_mask"][:, :args.max_length]
    text_and_candidates_ids.append(ids)
    text_and_candidates_masks.append(mask)
text_and_candidates_ids = torch.cat(text_and_candidates_ids, 0).unsqueeze(0)
text_and_candidates_ids = text_and_candidates_ids.to(args.device)
text_and_candidates_masks = torch.cat(text_and_candidates_masks, 0).unsqueeze(0)
text_and_candidates_masks = text_and_candidates_masks.to(args.device)
# inference
mode = torch.tensor([0]).to(args.device)
scores = torch.randn(1, len(args.scoring_methods), len(candidates))
scores = scores.to(args.device) # create random candidate scores
with torch.no_grad():
    output = summareranker_model(
        mode, 
        text_and_candidates_ids, 
        text_and_candidates_masks, 
        scores
    )
candidate_scores = output["overall_predictions"][0]
print("\nSummaReranker predicted scores:")
for j in range(len(candidates)):
    print("Candidate {} has score: {:.4f}".format(j, candidate_scores[j]))
best_idx = np.argmax(np.array(candidate_scores))
print("\nSummaReranker output summary is candidate #{}".format(best_idx))
print(candidates[best_idx])
