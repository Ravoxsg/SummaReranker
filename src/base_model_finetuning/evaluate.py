# Evaluate fine-tuned models

import argparse
import torch
import time

from utils import *
from data import *
from dataset import *
from dataset_trainer import *
from transfer_utils import *
from model import FTModel
from engine import validate



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--cuda', type = bool, default = True)
parser.add_argument('--debug', type = bool, default = False)
parser.add_argument('--debug_size', type = int, default = 30)

# data
parser.add_argument('--dataset', type = str, default = "reddit",
                    choices = ["cnndm", "xsum", "reddit"])
parser.add_argument('--val_max_size', type = int, default = 100000)
parser.add_argument('--check_data_pipe', type = bool, default = False)
parser.add_argument('--compute_r1s', type = bool, default = False)

# model
parser.add_argument('--model_type', type = str, default = "pegasus",
                    choices=["pegasus", "bart"])
parser.add_argument('--model', type = str, default = "google/pegasus-large",
                    choices=["google/pegasus-large", "facebook/bart-large"])
parser.add_argument('--hidden_size', type = int, default = 768)
parser.add_argument('--cache_dir', type = str, default = "../../../hf_models/pegasus-large/")
parser.add_argument('--load_model', type = bool, default = True)
parser.add_argument('--save_model_path', type = str, default = "ft_saved_models/reddit/pegasus_reddit_train_1/checkpoint-1200/pytorch_model.bin")

# evaluation
parser.add_argument('--inference_bs', type = int, default = 8)
parser.add_argument('--val_dataset', type = str, default = "test")

# metrics
# 1 - ROUGE
parser.add_argument('--eval_rouge', type = bool, default = True)
# 2 - BERTScore
parser.add_argument('--eval_bertscore', type = bool, default = False)
# 3 - BARTScore
parser.add_argument('--eval_bartscore', type = bool, default = False)
# 4 - Copying
parser.add_argument('--eval_ngram_copying', type = bool, default = False)
# 5 - Abstractiveness
parser.add_argument('--eval_new_ngram', type = bool, default = True)
parser.add_argument('--eval_target_abstractiveness_recall', type = bool, default = True)
# 6 - Overlap with source
parser.add_argument('--eval_rouge_text', type = bool, default = False)
# 7_stats
parser.add_argument('--check_correlation', type = bool, default = False)

# summary generation
parser.add_argument('--inference', type = bool, default = False)
parser.add_argument('--generation', type = bool, default = True)
parser.add_argument('--num_return_sequences', type = int, default = 1) # default: 1
parser.add_argument('--repetition_penalty', type = float, default = 1.0) # 1.0
parser.add_argument('--stemmer', type = bool, default = True)
parser.add_argument('--n_show_summaries', type = int, default = 1)

args = parser.parse_args()

dataset_names = ["cnndm", "xsum", "reddit"]
max_lengths = [1024, 512, 512]
max_summary_lengths = [128, 64, 64]
length_penalties_pegasus = [0.8, 0.8, 0.6]
length_penalties_bart = [0.8, 0.8, 1.0]
no_repeat_ngram_sizes_pegasus = [0, 3, 3]
no_repeat_ngram_sizes_bart = [0, 3, 3]
highlights = [True, False, False]
clean_ns = [True, False, False]

idx = dataset_names.index(args.dataset)
args.data_folder = "/data/mathieu/DATASETS/{}/data/".format(args.dataset)
args.max_length = max_lengths[idx]
args.max_summary_length = max_summary_lengths[idx]
if args.model_type == "pegasus":
    args.length_penalty = length_penalties_pegasus[idx]
    args.no_repeat_ngram_size = no_repeat_ngram_sizes_pegasus[idx]
    args.num_beams = 8
elif args.model_type == "bart":
    args.length_penalty = length_penalties_bart[idx]
    args.no_repeat_ngram_size = no_repeat_ngram_sizes_bart[idx]
    args.num_beams = 5
args.highlights = highlights[idx]
args.clean_n = clean_ns[idx]

print("*"*50)
print(args)



def main(args):
    # seed
    seed_everything(args.seed)

    # device
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    args.device = device
    print("\nUsing device {}".format(device))

    # data
    val_data = load_data(args.val_dataset, args, individual_txt = args.highlights)

    # tokenizer
    tokenizer = build_tokenizer(args)

    # datasets
    datasets = []
    for x in [("val", val_data)]:
        mode, data = x
        texts, summaries = data
        print(len(texts), len(summaries))
        if args.debug:
            texts = texts[:args.debug_size]
            summaries = summaries[:args.debug_size]
        texts = texts[:args.val_max_size]
        summaries = summaries[:args.val_max_size]
        dataset = InferenceFTDataset(mode, tokenizer, texts, summaries, args)
        datasets.append(dataset)
        print("There are {} {} batches".format(int(len(dataset.texts) / args.inference_bs), mode))
    val_dataset = datasets[0]

    # data loader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.inference_bs, shuffle = False)

    # check data pipe
    if args.check_data_pipe:
        check_data_pipe([val_loader])

    # model
    base_model = build_model(args)
    model = FTModel(base_model, args)
    #model = base_model
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nThe model has {} trainable parameters".format(n_params))
    model = model.to(device)
    if args.load_model:
        print("loading the weights: {}".format(args.save_model_path))
        model.load_state_dict(torch.load(args.save_model_path))
        print("loaded the model weights!")

    # training
    validate("val", val_loader, [], tokenizer, model, device, args)


if __name__ == '__main__':

    main(args)
