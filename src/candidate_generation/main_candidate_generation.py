# Generate summary candidates with the fine-tuned models.

import time
import argparse
import sys

sys.path.append("/data/mathieu/SummaReranker/src/") # todo: change to your folder path

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
parser.add_argument('--debug_size', type = int, default = 10)

# data
parser.add_argument('--dataset', type=str, default = "reddit", 
                    choices= ["cnndm", "xsum", "reddit"])

# model
parser.add_argument('--model_type', type = str, default = "pegasus",
                    choices=["pegasus", "bart"])
parser.add_argument('--model', type = str, default = "google/pegasus-cnn_dailymail",
                    choices = ["google/pegasus-large", "google/pegasus-cnn_dailymail", "google/pegasus-xsum",
                    "facebook/bart-large", "facebook/bart-large-cnn", "facebook/bart-large-xsum"])
parser.add_argument('--model_name', type=str, default = "pegasus_cnndm",
                    choices = ["pegasus_unsupervised", "bart_unsupervised",
                    "pegasus_cnndm_first_half_shuffled_1", "pegasus_cnndm_second_half_shuffled_1", "pegasus_cnndm",
                    "bart_cnndm_first_half_shuffled_1", "bart_cnndm_second_half_shuffled_1", "bart_cnndm",
                    "pegasus_xsum_first_half_shuffled_1", "pegasus_xsum_second_half_shuffled_1", "pegasus_xsum",
                    "bart_xsum_first_half_shuffled_1", "bart_xsum_second_half_shuffled_1", "bart_xsum",
                    "pegasus_reddit_first_half_shuffled_1", "pegasus_reddit_second_half_shuffled_1", "pegasus_reddit_train_1",
                    "bart_reddit_first_half_shuffled_1", "bart_reddit_second_half_shuffled_1", "bart_reddit_train_1"])
parser.add_argument('--hidden_size', type = int, default = 768) # 768 / 1024`
parser.add_argument('--cache_dir', type = str,
                    default = "../../../hf_models/pegasus-large-cnndm/")
parser.add_argument('--load_model', type = bool, default = False)
parser.add_argument('--load_model_path', type = str,
                    default = "../base_model_finetuning/ft_saved_models/reddit/pegasus_reddit_train_1/checkpoint-5/pytorch_model.bin") # todo: change to where you saved the finetuned checkpoint

# summary generation
parser.add_argument('--val_dataset', type=str, default = "test",
                    choices = ["train", "first_half_train_shuffled", "second_half_train_shuffled", "val", "test"])
parser.add_argument('--max_val_size', type = int, default = 1000000)
parser.add_argument('--inference_bs', type = int, default = 2) 
parser.add_argument('--save_summaries', type = bool, default = True)
parser.add_argument('--generation_method', type = str, default = "beam_search",
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
max_lengths = [1024, 512, 512]
max_summary_lengths = [128, 64, 128]
clean_ns = [True, False, False]
length_penalties_pegasus = [0.8, 0.8, 0.6]
length_penalties_bart = [0.8, 0.8, 1.0]
repetition_penalties = [1.0, 1.0, 1.0]
no_repeat_ngram_sizes = [0, 3, 3]

idx = dataset_names.index(args.dataset)

args.highlights = highlights[idx]
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

    if not(os.path.isdir("../../summaries/")):
        os.makedirs("../../summaries/")
    if not(os.path.isdir("../../summaries/{}/".format(args.dataset))):
        os.makedirs("../../summaries/{}/".format(args.dataset))
    if not(os.path.isdir("../../summaries/{}/{}/".format(args.dataset, args.val_dataset))):
        os.makedirs("../../summaries/{}/{}/".format(args.dataset, args.val_dataset))
    if not(os.path.isdir("../../summaries/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method))):
        os.makedirs("../../summaries/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method))

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
    mode = "val"
    texts, summaries = val_data
    print("Data size", len(texts), len(summaries))

    texts = texts[:args.max_val_size]
    summaries = summaries[:args.max_val_size]
    print("Data size after truncation", len(texts), len(summaries))
    if args.debug:
        texts = texts[:args.debug_size]
        summaries = summaries[:args.debug_size]
    val_dataset = Dataset(mode, tokenizer, texts, summaries, args)
    print("Total size of dataset: {}".format(len(texts)))

    # data loader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.inference_bs, shuffle = False)

    # model
    model = build_model(args)
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

    print(base_results[0])

    # export
    num_candidates = len(val_summaries[0])
    if args.save_summaries:
        path = "../../summaries/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method)
        with open(path + "{}_texts_{}_beams_{}.pkl".format(args.val_dataset, len(val_texts), num_candidates), "wb") as f:
            pickle.dump(val_texts, f)
        with open(path + "{}_summaries_{}_{}_beams_{}.pkl".format(args.val_dataset, args.model_name, len(val_texts), num_candidates), "wb") as f:
            pickle.dump(val_summaries, f)
        with open(path + "{}_labels_{}_beams_{}.pkl".format(args.val_dataset, len(val_texts), num_candidates), "wb") as f:
            pickle.dump(val_labels, f)
        print("saved generated summaries!", path)


if __name__ == '__main__':

    main(args)
