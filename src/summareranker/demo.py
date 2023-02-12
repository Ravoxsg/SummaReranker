# Complete SummaReranker generation inference pipeline in a single small script!

import sys
sys.path.append("/data/mathieu/SummaReranker/src/")
import numpy as np
import torch
import argparse
from datasets import load_dataset
from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration, RobertaTokenizerFast, RobertaModel

from common.utils import seed_everything
from model import ModelMultitaskBinary


parser = argparse.ArgumentParser()
args = parser.parse_args()
args.device = torch.device("cuda")
args.generation_methods = ["diverse_beam_search"]
args.num_beams = 15
args.scoring_methods = ["rouge_1", "rouge_2", "rouge_l"]
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
args.max_source_length = 384
args.max_summary_length = 128

# seed
seed_everything(42)

# data
dataset_name = "ccdv/cnn_dailymail"
subset = "test"
text_key = "article"
dataset = load_dataset(dataset_name, cache_dir="/data/mathieu/hf_datasets/")
dataset = dataset[subset]
texts = dataset[text_key]
p = np.random.permutation(len(texts))
texts = [texts[x] for x in p]
text = texts[0]
text = text.replace("\n", " ")
print("\nSource document:")
print(text)

# base model
base_model_name = "google/pegasus-cnndm"
base_tokenizer = PegasusTokenizerFast.from_pretrained(base_model_name, cache_dir="/data/mathieu/hf_models/pegasus-large-cnndm/")
base_model = PegasusForConditionalGeneration.from_pretrained(base_model_name, cache_dir="/data/mathieu/hf_models/pegasus-large-cnndm/")
base_model = base_model.cuda()

# candidates
tok_text = base_tokenizer(text, return_tensors="pt", padding="max_length", max_length=1024)
tok_text["input_ids"] = tok_text["input_ids"][:, :1024]
tok_text["attention_mask"] = tok_text["attention_mask"][:, :1024]
generated = base_model.generate(
    input_ids=tok_text["input_ids"].cuda(),
    attention_mask=tok_text["attention_mask"].cuda(),
    num_beams=15,
    num_beam_groups=15,
    num_return_sequences=15,
    diversity_penalty=1.0,
    repetition_penalty=1.0,
    length_penalty=0.8,
    no_repeat_ngram_size=3
)
candidates = base_tokenizer.batch_decode(generated, skip_special_tokens=True)
print("\nSummary candidates:")
for j in range(len(candidates)):
    print("Candidate {}:".format(j))
    print(candidates[j])

# SummaReranker
# model
model_name = "roberta-large"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name, cache_dir="/data/mathieu/hf_models/roberta-large/")
model = RobertaModel.from_pretrained(model_name, cache_dir="/data/mathieu/hf_models/roberta-large/")
model = model.cuda()
summareranker_model = ModelMultitaskBinary(model, tokenizer, args)
summareranker_model = summareranker_model.cuda()
summareranker_model_path = "/data/mathieu/2nd_stage_summarization/4_supervised_multitask_reranking/saved_models/cnndm/multitask_3_tasks_ablation_8/checkpoint-12500/pytorch_model.bin"
summareranker_model.load_state_dict(torch.load(summareranker_model_path))
# prepare the data
text_inputs = self.tokenizer(text, return_tensors="pt", max_length=self.args.max_source_length, padding='max_length')
text_inputs["input_ids"] = text_inputs["input_ids"][:, :self.args.max_source_length]
text_inputs["attention_mask"] = text_inputs["attention_mask"][:, :self.args.max_source_length]
text_and_candidates_ids, text_and_candidates_masks = [], []
for j in range(len(candidates)):
    candidate = candidates[j]
    candidate_inputs = self.tokenizer(candidate, return_tensors="pt", max_length=self.args.max_summary_length, padding='max_length')
    candidate_inputs["input_ids"] = candidate_inputs["input_ids"][:, :self.args.max_summary_length]
    candidate_inputs["attention_mask"] = candidate_inputs["attention_mask"][:, :self.args.max_summary_length]
    ids = torch.cat((text_inputs["input_ids"], candidate_inputs["input_ids"]), 1)
    mask = torch.cat((text_inputs["attention_mask"], candidate_inputs["attention_mask"]), 1)
    text_and_candidates_ids.append(ids)
    text_and_candidates_masks.append(mask)
text_and_candidates_ids = torch.cat(text_and_candidates_ids, 0)
text_and_candidates_masks = torch.cat(text_and_candidates_masks, 0)
print(ids.shape, masks.shape)
# inference
mode = "val"
scores = torch.randn(len(text_and_candidates_ids), len(args.scoring_methods)) # create random candidate scores
output = model(mode, text_and_candidates_ids, text_and_candidates_mask, scores)
candidate_scores = output["overall_predictions"]
print(candidate_scores)