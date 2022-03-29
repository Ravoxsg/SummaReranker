# Generate summary candidates with the fine-tuned models.

import time
import argparse
import sys

sys.path.append("/data/mathieu/CODE_RELEASES/SummaReranker/src/")

from transformers import AutoTokenizer, AutoModel

from common.utils import *
from common.evaluation import *
from common.data import load_data
from dataset import *
from model import *
from engine import *
from model_utils import *



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--cuda', type = bool, default = True)
parser.add_argument('--debug', type = bool, default = False)
parser.add_argument('--debug_size', type = int, default = 30)

# data
parser.add_argument('--dataset', type=str, default = "reddit", 
                    choices= ["cnndm", "xsum", "reddit"]) 
parser.add_argument('--data_folder', type = str, default = "/data/mathieu/DATASETS/RedditTIFU/data/") 

# model
parser.add_argument('--model_type', type = str, default = "pegasus") # in ["t5", "pegasus", "bart"]
parser.add_argument('--model', type = str, default = "google/pegasus-large",
                    choices = ["google/pegasus-large", "google/pegasus-cnn_dailymail", "google/pegasus-xsum",
                    "facebook/bart-large", "facebook/bart-large-cnn", "facebook/bart-large-xsum"])
parser.add_argument('--model_name', type=str, default = "pegasus_reddit_train_1",
                    choices = ["pegasus_cnndm", "bart_cnndm", "pegasus_xsum", "bart_xsum", 
                    "pegasus_reddit_train_1", "bart_reddit"])
parser.add_argument('--hidden_size', type = int, default = 768) # 768 / 1024`
parser.add_argument('--cache_dir', type = str, default = "../../hf_models/pegasus-large-reddit/") 
parser.add_argument('--load_model', type = bool, default = True)
parser.add_argument('--load_model_path', type = str, default = "/data/mathieu/2nd_stage_summarization/1_base_finetuning/ft_saved_models/pegasus_reddit_train_1/checkpoint-1250/pytorch_model.bin")
parser.add_argument('--ft_model', type = bool, default = True)

# summary generation
parser.add_argument('--val_dataset', type=str, default = "small_val",
                    choices = ["small_val", "val", "test"]) 
parser.add_argument('--val_size', type=int, default = 300) 
parser.add_argument('--inference_bs', type = int, default = 2) 
parser.add_argument('--save_summaries', type = bool, default = False)
parser.add_argument('--save_summaries_path', type = str, default = "../summaries/Reddit/2_diverse_beam_search/")
parser.add_argument('--generation_method', type = str, default = "diverse_beam_search",
                    choices = ["beam_search", "diverse_beam_search", "top_p_sampling", "top_k_sampling"])
parser.add_argument('--num_return_sequences', type = int, default = 15) # default: 15
parser.add_argument('--num_beams', type = int, default = 15) # for beam search
parser.add_argument('--num_beam_groups', type = int, default = 15) # for diverse beam search
parser.add_argument('--diversity_penalty', type = float, default = 1.0) # for diverse beam search
parser.add_argument('--top_p', type = float, default = 0.95) # for top-p sampling
parser.add_argument('--top_k', type = int, default = 50) # for top-k sampling
parser.add_argument('--stemmer', type = bool, default = True)

# metrics 
parser.add_argument('--eval_rouge', type = bool, default = True)
parser.add_argument('--eval_bertscore', type = bool, default = False)
parser.add_argument('--eval_bartscore', type = bool, default = False)
parser.add_argument('--eval_new_ngram', type = bool, default = True)
parser.add_argument('--eval_rouge_text', type = bool, default = False)

args = parser.parse_args()

dataset_names = ["cnndm", "xsum", "reddit"]
highlights = [True, False, False]
val_data_sizes = [13368, 11332, 4213]
test_data_sizes = [11490, 11334, 4222]
max_lengths = [384, 448, 384]
max_summary_lengths = [128, 64, 128]
clean_ns = [True, False, False]
length_penalties_pegasus = [0.8, 0.8, 0.6]
length_penalties_bart = [0.8, 0.8, 1.0]
repetition_penalties = [1.0, 1.0, 1.0]
no_repeat_ngram_sizes = [0, 3, 3]

idx = dataset_names.index(args.dataset)

args.highlights = highlights[idx]
if args.val_dataset == "small_val":
    args.val_data_size = 300
elif args.val_dataset == "val":
    args.val_data_size = val_data_sizes[idx]
elif args.val_dataset == "test":
    args.val_data_size = test_data_sizes[idx]
args.test_data_size = test_data_sizes[idx]
args.max_length = max_lengths[idx]
args.max_summary_length = max_summary_lengths[idx]
args.clean_n = clean_ns[idx]
if args.model_type == "pegasus":
    args.length_penalty = length_penalties_pegasus[idx]
elif args.model_type == "bart":
    args.length_penalty = length_penalties_bart[idx]
args.repetition_penalty = repetition_penalties[idx]
args.no_repeat_ngram_size = no_repeat_ngram_sizes[idx]

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
    for x in [(args.val_dataset, val_data)]:
        mode, data = x
        texts, summaries = data
        print(len(texts), len(summaries))
        texts = texts[:args.val_data_size]
        summaries = summaries[:args.val_data_size]
        print(len(texts), len(summaries))
        if args.debug:
            texts = texts[:args.debug_size]
            summaries = summaries[:args.debug_size]
        dataset = Dataset(mode, tokenizer, texts, summaries, args)
        print("Total size of dataset: {}".format(len(texts)))
        datasets.append(dataset)
    val_dataset = datasets[0]

    # data loader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.inference_bs, shuffle = False)

    # model
    model = build_model(args)
    if args.ft_model:
        model = FTModel(model, args)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nThe model has {} trainable parameters".format(n_params))
    model = model.to(device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model_path))
        print("Loaded the model weights!", args.load_model_path)

    # summary generation
    val_texts, val_summaries, val_labels = get_summaries(tokenizer, val_loader, model, device, args)

    # evaluation
    base_results = [val_summaries[i][0] for i in range(len(val_summaries))]
    print("*"*100)
    print("\nTop beam:")
    overall_eval(val_texts, base_results, val_labels, args)

    # export
    num_candidates = len(val_summaries[0])
    if args.save_summaries:
        with open(args.save_summaries_path + "{}/".format(args.val_dataset) + "{}_texts_{}_beams_{}.pkl".format(args.val_dataset, len(val_texts), num_candidates), "wb") as f:
            pickle.dump(val_texts, f)
        with open(args.save_summaries_path + "{}/".format(args.val_dataset) + "{}_summaries_{}_{}_beams_{}.pkl".format(args.val_dataset, args.model_name, len(val_texts), num_candidates), "wb") as f:
            pickle.dump(val_summaries, f)
        with open(args.save_summaries_path + "{}/".format(args.val_dataset) + "{}_labels_{}_beams_{}.pkl".format(args.val_dataset, len(val_texts), num_candidates), "wb") as f:
            pickle.dump(val_labels, f)
        print("saved generated summaries!", args.save_summaries_path + "{}/".format(args.val_dataset))



if __name__ == '__main__':

    main(args)
