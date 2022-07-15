# Score each summary candidate according to a specified metric.

import argparse
import pickle
import sys
import gc

sys.path.append("/data/mathieu/SummaReranker/src/") # todo: change to your folder path

from time import time 
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from rouge_score import rouge_scorer
from datasets import load_metric
from bert_score import score as bertscore_score

from common.utils import *
from common.bart_score import BARTScorer
from common.evaluation import overall_eval



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)

# data
parser.add_argument('--dataset', type=str, default = "reddit", 
                    choices= ["cnndm", "xsum", "reddit"])
parser.add_argument('--val_dataset', type = str, default = "val",
                    choices = ["train", "first_half_train_shuffled", "second_half_train_shuffled", "val", "test"])
parser.add_argument('--generation_method', type = str, default = "diverse_beam_search",
                    choices = ["beam_search", "diverse_beam_search", "top_p_sampling", "top_k_sampling"])
parser.add_argument('--val_size', type = int, default = -1)

# model
parser.add_argument('--model_name', type = str, default = "pegasus_unsupervised",
                    choices = ["pegasus_unsupervised", "bart_unsupervised",
                    "pegasus_cnndm_first_half_shuffled_1", "pegasus_cnndm_second_half_shuffled_1", "pegasus_cnndm", 
                    "bart_cnndm_first_half_shuffled_1", "bart_cnndm_second_half_shuffled_1", "bart_cnndm",
                    "pegasus_xsum_first_half_shuffled_1", "pegasus_xsum_second_half_shuffled_1", "pegasus_xsum", 
                    "bart_xsum_first_half_shuffled_1", "bart_xsum_second_half_shuffled_1", "bart_xsum", 
                    "pegasus_reddit_first_half_shuffled_1", "pegasus_reddit_second_half_shuffled_1", "pegasus_reddit_train_1", 
                    "bart_reddit_first_half_shuffled_1", "bart_reddit_second_half_shuffled_1", "bart_reddit"])
parser.add_argument('--num_candidates', type = int, default = 15)

# METRIC
parser.add_argument('--label_metric', type = str, default = "rouge_1",
                    choices = ["mean_rouge", "rouge_1", "rouge_2", "rouge_l", "bertscore", "bartscore"])

# evaluation
parser.add_argument('--stemmer', type = bool, default = True)

# export
parser.add_argument('--save_scores', type = bool, default = False)

# metrics
parser.add_argument('--eval_top_candidate', type = bool, default = True)
parser.add_argument('--eval_oracle', type = bool, default = True)
parser.add_argument('--eval_rouge', type = bool, default = True)
parser.add_argument('--eval_bertscore', type = bool, default = False)
parser.add_argument('--eval_bartscore', type = bool, default = False)
parser.add_argument('--eval_new_ngram', type = bool, default = False)
parser.add_argument('--eval_rouge_text', type = bool, default = False)

args = parser.parse_args()

dataset_names = ["cnndm", "xsum", "reddit"]
highlights = [True, False, False]
val_data_sizes = [13368, 11332, 4213]
test_data_sizes = [11490, 11334, 4222]
clean_ns = [True, False, False]

idx = dataset_names.index(args.dataset)

args.highlights = highlights[idx]
if args.val_size < 0:
    if args.val_dataset == "val":
        args.val_size = val_data_sizes[idx]
    elif args.val_dataset == "test":
        args.val_size = test_data_sizes[idx]
args.test_data_size = test_data_sizes[idx]
args.clean_n = clean_ns[idx]

print("*"*50)
print(args)



def main(args):
    # seed
    seed_everything(args.seed)

    if not(os.path.isdir("../../scored_summaries/")):
        os.makedirs("../../scored_summaries/")
    if not(os.path.isdir("../../scored_summaries/{}/".format(args.dataset))):
        os.makedirs("../../scored_summaries/{}/".format(args.dataset))
    if not(os.path.isdir("../../scored_summaries/{}/{}/".format(args.dataset, args.val_dataset))):
        os.makedirs("../../scored_summaries/{}/{}/".format(args.dataset, args.val_dataset))
    if not(os.path.isdir("../../scored_summaries/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method))):
        os.makedirs("../../scored_summaries/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method))
    if not(os.path.isdir("../../scored_summaries/{}/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method, args.label_metric))):
        os.makedirs("../../scored_summaries/{}/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method, args.label_metric))

    # path
    path = "../../summaries/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method)
    # load summaries
    summaries_path = path + "{}_summaries_{}_{}_beams_{}.pkl".format(args.val_dataset, args.model_name, args.val_size, args.num_candidates)
    with open(summaries_path, "rb") as f:
        summaries = pickle.load(f)
    print("Loaded {} summaries".format(len(summaries)))
    # load labels
    labels_path = path + "{}_labels_{}_beams_{}.pkl".format(args.val_dataset, args.val_size, args.num_candidates)
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
    print("Loaded {} labels".format(len(labels)))

    # score summaries against the labels
    print("\nSCORING SUMMARIES WITH: {}".format(args.label_metric))

    # init
    if "rouge" in args.label_metric:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer = args.stemmer)
        scores = []
    elif args.label_metric == "bertscore":
        all_summaries = []
        all_labels = []
        for j in range(args.num_candidates):
            all_summaries.append([])
    elif args.label_metric == "bleurt":
        metric = load_metric('bleurt', keep_in_memory = True)
    elif args.label_metric == "bartscore":
        all_summaries = []
        all_labels = []
        for j in range(args.num_candidates):
            all_summaries.append([])
        bart_scorer = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
        scores = []

    t1 = time()
    # loop
    for i in tqdm(range(len(summaries))):
        summaries_i = summaries[i]
        label = labels[i]
        if "rouge" in args.label_metric:
            scores_i = get_rouge_scores(label, summaries_i, scorer, args)
            scores.append(scores_i)
        elif args.label_metric in  ["bertscore", "bartscore"]:
            all_labels.append(label)
            for j in range(len(summaries_i)):
                all_summaries[j].append(summaries_i[j])
        elif args.label_metric == "bleurt":
            metric.add_batch(predictions=summaries_i, references=[label] * len(summaries_i))

    # conclusion
    if args.label_metric == "bertscore":
        all_f1 = []
        for j in range(len(all_summaries)):
            print(j, len(all_summaries[j]))
            _, _, f1 = bertscore_score(all_summaries[j], all_labels, lang='en', verbose=True, batch_size=16)
            all_f1.append(f1)
            gc.collect()
        scores = [[all_f1[j][i].item() for j in range(len(all_f1))] for i in range(len(all_f1[0]))]
    elif args.label_metric == "bleurt":
        score = metric.compute()
        bleurt_scores = score["scores"]
        print(len(bleurt_scores))
        scores = []
        for i in range(len(summaries)):
            scores.append(bleurt_scores[(i*args.num_candidates):((i+1)*args.num_candidates)])
    elif args.label_metric == "bartscore":
        all_bartscores = []
        for j in range(len(all_summaries)):
            print(j, len(all_summaries[j]))
            bartscores = bart_scorer.score(all_labels, all_summaries[j], batch_size=16)
            all_bartscores.append(bartscores)
        scores = [[all_bartscores[j][i] for j in range(len(all_bartscores))] for i in range(len(all_bartscores[0]))]
    t2 = time()
    print("Time to get the scores: {:.4f}".format(t2-t1))
    print(len(scores), len(scores[0]))
    print(type(scores[0]))
    print(scores[0])
    scored_summaries = [[summaries[i], scores[i]] for i in range(len(summaries))]
    top_scores = [scores[i][0] for i in range(len(summaries))]
    oracle_scores = [np.max(scores[i]) for i in range(len(summaries))]

    print(len(scored_summaries))
    print("Mean score (top beam): {:.4f}".format(np.mean(np.array(top_scores))))
    print("ORACLE score: {:.4f}".format(np.mean(np.array(oracle_scores))))

    if args.save_scores:
        new_path = "../../scored_summaries/{}/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method, args.label_metric)
        save_path = new_path + "{}_scored_summaries_{}_{}_beams_{}.pkl".format(args.val_dataset, args.model_name, args.val_size, args.num_candidates)
        with open(save_path, "wb") as f:
            pickle.dump(scored_summaries, f)
            print("saved new data!", save_path)

    if args.eval_top_candidate:
        val_summaries = [summaries[i][0] for i in range(len(summaries))]
        print("\n\n")
        print("*"*50)
        print("Top candidate evaluation:")
        overall_eval(None, val_summaries, labels, args)

    if args.eval_oracle:
        val_summaries = [summaries[i][np.argmax(scores[i])] for i in range(len(summaries))]
        print("\n\n")
        print("*"*50)
        print("Oracle evaluation:")
        overall_eval(None, val_summaries, labels, args)


def get_rouge_scores(label, summaries_i, scorer, args):
    scores_i = []
    for j in range(len(summaries_i)):
        summary = summaries_i[j]
        if args.clean_n:
            summary = summary.replace("<n>", " ")
        if args.highlights:
            summary = "\n".join(sent_tokenize(summary))
        rouge_scores = scorer.score(label, summary)
        r1 = 100 * rouge_scores["rouge1"].fmeasure
        r2 = 100 * rouge_scores["rouge2"].fmeasure
        rl = 100 * rouge_scores["rougeLsum"].fmeasure
        if args.label_metric == "mean_rouge":
            score = (r1 + r2 + rl) / 3
        elif args.label_metric == "rouge_1":
            score = r1
        elif args.label_metric == "rouge_2":
            score = r2
        elif args.label_metric == "rouge_l":
            score = rl
        scores_i.append(score)

    return scores_i



if __name__ == '__main__':

    main(args)
