# Evaluate the performance of a trained re-ranker.

import argparse
import sys
import time
import torch

sys.path.append("/data/mathieu/SummaReranker/src/")

from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaModel

from common.utils import seed_everything
from common.evaluation import *
from common.data_scored import load_data
from utils import *
from dataset import MultitaskRerankingDataset
from model import ModelMultitaskBinary



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default = 42)
parser.add_argument('--cuda', type=bool, default = True)

# data
parser.add_argument('--dataset', type=str, default = "reddit", 
                    choices= ["cnndm", "xsum", "reddit"])
parser.add_argument('--generation_methods_str', type=str, default = "diverse_beam_search")
parser.add_argument('--scoring_methods_str', type=str, default = "rouge_1+rouge_2+rouge_l")
parser.add_argument('--sep_symbol', type=str, default = "[SEP]")
parser.add_argument('--val_dataset', type=str, default = "val", choices = ["val", "test"])
parser.add_argument('--max_val_size', type=int, default = 300)

# base model
parser.add_argument('--model_name', type = str, default = "pegasus_unsupervised",
                    choices = ["pegasus_unsupervised", "pegasus_cnndm", "bart_cnndm",
                    "pegasus_xsum", "bart_xsum", "pegasus_reddit_train_1", "bart_reddit"])
parser.add_argument('--num_beams', type=int, default = 15)

# model
parser.add_argument('--sharp_pos', type=bool, default = False)
parser.add_argument('--model', type=str, default = "roberta-large")  
parser.add_argument('--cache_dir', type=str, default = "../../../hf_models/roberta-large/")
parser.add_argument('--hidden_size', type=int, default = 1024) 
parser.add_argument('--non_linear_repres', type=bool, default = True)
parser.add_argument('--use_shared_bottom', type = bool, default = True)
parser.add_argument('--bottom_hidden_size', type = int, default = 1024)
parser.add_argument('--num_experts', type=int, default = 6)
parser.add_argument('--k', type=int, default = 3)
parser.add_argument('--use_aux_loss', type = bool, default = False)
parser.add_argument('--expert_hidden_size', type = int, default = 1024)
parser.add_argument('--tower_hidden_size', type = int, default = 1024)
parser.add_argument('--load_model', type=bool, default = True)
parser.add_argument('--load_model_path', type=str, default = "/data/mathieu/2nd_stage_summarization/4_supervised_multitask_reranking/saved_models/reddit/multitask_3_tasks_ablation_5/checkpoint-1000/pytorch_model.bin")
# todo: change the path to where you saved the SummaReranker checkpoint!

# optimization
parser.add_argument('--inference_bs', type=int, default = 60)

# generation
parser.add_argument('--stemmer', type = bool, default = True)

# metrics
parser.add_argument('--eval_rouge', type = bool, default = True)
parser.add_argument('--eval_bertscore', type = bool, default = True)
parser.add_argument('--eval_bartscore', type = bool, default = True)
parser.add_argument('--eval_new_ngram', type = bool, default = True)
parser.add_argument('--eval_rouge_source', type = bool, default = False)

args = parser.parse_args()
args.generation_methods = args.generation_methods_str.split("+")
args.scoring_methods = args.scoring_methods_str.split("+")
args.n_tasks = len(args.scoring_methods)

dataset_names = ["cnndm", "xsum", "reddit"]
highlights = [True, False, False]
val_sizes = [13368, 11332, 4213]
test_sizes = [11490, 11334, 4222]
max_lengths = [384, 448, 384]
max_summary_lengths = [128, 64, 128]
clean_ns = [True, False, False]

idx = dataset_names.index(args.dataset)

args.highlights = highlights[idx]
if args.val_dataset == "val":
    args.val_size = val_sizes[idx]
elif args.val_dataset == "test":
    args.val_size = test_sizes[idx]
args.max_length = max_lengths[idx]
args.max_summary_length = max_summary_lengths[idx]
args.clean_n = clean_ns[idx]

print("*" * 50)
print(args)



def main(args):
    # seed
    seed_everything(args.seed)

    # device
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    args.device = device
    print("Using device: {}".format(device))

    # tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model, cache_dir = args.cache_dir)

    # data
    set = args.val_dataset
    size = args.val_size
    texts, summaries, scored_summaries = load_data(set, size, args, individual_txt = args.highlights)
    print("loaded new data!", len(texts), len(summaries), len(scored_summaries), len(scored_summaries[0]),
          len(scored_summaries[0][0]), len(scored_summaries[0][1]))
    texts = texts[:len(scored_summaries)]
    summaries = summaries[:len(summaries)]
    p = np.random.permutation(len(texts))
    p = p[:args.val_size]
    texts = [texts[i] for i in p]
    summaries = [summaries[i] for i in p]
    scored_summaries = [scored_summaries[i] for i in p]

    # dataset
    mode = "val"
    val_dataset = MultitaskRerankingDataset(mode, tokenizer, texts, scored_summaries, summaries, args)
    val_dataset.texts = val_dataset.texts[:args.max_val_size]
    val_dataset.scored_summaries = val_dataset.scored_summaries[:args.max_val_size]
    val_dataset.labels = val_dataset.labels[:args.max_val_size]
    print("There are {} {} batches".format(int(len(val_dataset.texts) / args.inference_bs), set))
    
    # data loader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.inference_bs, shuffle = False)

    # model
    pretrained_model = RobertaModel.from_pretrained(args.model, cache_dir = args.cache_dir)
    n_params = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    print("\nThe base LM has {} trainable parameters".format(n_params))
    model = ModelMultitaskBinary(pretrained_model, tokenizer, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nThe model has {} trainable parameters".format(n_params))
    model = model.to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model_path))
        print("Loaded the model weights!", args.load_model_path)

    # inference
    val_texts = []
    val_labels = []
    val_preds_idx = []
    val_predictions = []
    val_overall_predictions = []
    for i, batch in tqdm(enumerate(val_loader)):
        model.zero_grad()

        mode = batch["mode"]
        batch_texts = batch["text"]
        val_texts += batch_texts
        batch_labels = batch["label"]
        val_labels += batch_labels

        text_and_summaries_ids = batch["text_and_summaries_input_ids"].to(device)
        text_and_summaries_mask = batch["text_and_summaries_attn_mask"].to(device)
        scores = batch["scores"]

        with torch.no_grad():
            output = model(mode, text_and_summaries_ids, text_and_summaries_mask, scores)
            predictions_idx = output["total_predictions_idx"]
            val_preds_idx += predictions_idx
            val_predictions.append(output["prediction_sum"].item()) 
            val_overall_predictions += output["overall_predictions"]
    print("# texts: {}, # summaries: {}, # preds idx: {}, predictions: {}".format(len(val_texts), len(val_labels), len(val_preds_idx), len(val_predictions)))
    print("Mean predictions: {:.4f}".format(np.mean(np.array(val_predictions))))

    val_preds = []
    for i in range(len(val_preds_idx)):
        val_preds.append(scored_summaries[i][0][val_preds_idx[i]])

    # evaluation
    overall_eval(val_texts, val_preds, val_labels, args)



if __name__ == '__main__':
    main(args)
