# Separate the training sets into 2 parts

import argparse
import sys
import numpy as np
import pickle

sys.path.append("/data/mathieu/2nd_stage_summarization/")

from tqdm import tqdm
from shutil import copyfile

from common.utils import seed_everything



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--dataset', type=str, default = "reddit",
                    choices=["cnndm", "xsum", "reddit"])

args = parser.parse_args()

dataset_names = ["cnndm", "xsum", "reddit"]
threshs = [143000, 102000, 17000]
highlights = [True, False, False]

idx = dataset_names.index(args.dataset)
args.data_folder = "../../data/{}/".format(args.dataset)
args.thresh = threshs[idx]
args.individual_files = highlights[idx]

print("*"*50)
print(args)



def main(args):
    # seed
    seed_everything(args.seed)

    # load full training
    train_summaries, train_texts = [], []
    with open(args.data_folder + "train_summary.txt", "rb") as f:
        for l in f.readlines():
            train_summaries.append(l)
    with open(args.data_folder + "train_text.txt", "rb") as f:
        for l in f.readlines():
            train_texts.append(l)
    print(len(train_summaries), len(train_texts))

    # shuffle
    p = np.random.permutation(len(train_texts))
    print(p[:10])
    with open("dataset_permutations/{}_train_permutation.pkl".format(args.dataset), "wb") as f:
        pickle.dump(p, f)
        print("saved permutation!")
    train_summaries = [train_summaries[i] for i in p]
    train_texts = [train_texts[i] for i in p]
    print("permuted the training set!")
    p_to_normal = {}
    for i in range(len(p)):
        p_to_normal[p[i]] = i

    # 1st half - full files
    first_half_summaries = train_summaries[:args.thresh]
    first_half_texts = train_texts[:args.thresh]
    print(len(first_half_summaries), len(first_half_texts))
    with open(args.data_folder + "first_half_train_shuffled_summary.txt", "wb") as f:
        for l in first_half_summaries:
            f.write(l)
    with open(args.data_folder + "first_half_train_shuffled_text.txt", "wb") as f:
        for l in first_half_texts:
            f.write(l)

    # 2nd half - full files
    second_half_summaries = train_summaries[args.thresh:]
    second_half_texts = train_texts[args.thresh:]
    print(len(second_half_summaries), len(second_half_texts))
    with open(args.data_folder + "second_half_train_shuffled_summary.txt", "wb") as f:
        for l in second_half_summaries:
            f.write(l)
    with open(args.data_folder + "second_half_train_shuffled_text.txt", "wb") as f:
        for l in second_half_texts:
            f.write(l)
            
    # individual files
    if args.individual_files:
        docs = ["summary", "text"]
        for doc in docs:
            path = args.data_folder + "train/{}/".format(doc)
            print(path)
            idx_first = 0
            idx_second = 0
            for i in tqdm(range(len(p))):
                src_path = path + "train_{}_{}.txt".format(doc, p[i])
                if i < args.thresh:
                    dst_path = args.data_folder + "first_half_train_shuffled/{}/first_half_train_shuffled_{}_{}.txt".format(
                        doc, doc, idx_first)
                    idx_first += 1
                else:
                    dst_path = args.data_folder + "second_half_train_shuffled/{}/second_half_train_shuffled_{}_{}.txt".format(
                        doc, doc, idx_second)
                    idx_second += 1
                copyfile(src_path, dst_path)



if __name__ == '__main__':
    main(args)




