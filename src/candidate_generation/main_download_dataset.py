# Generate summary candidates with the fine-tuned models.

import time
import os
import numpy as np
import random
import argparse
import sys
import datasets

from tqdm import tqdm

sys.path.append("/data/mathieu/SummaReranker/src/") # todo: change to your folder path



parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)

# data
parser.add_argument('--dataset', type=str, default = "reddit",
                    choices= ["cnndm", "xsum", "reddit"])

args = parser.parse_args()

dataset_keys = ["cnndm", "xsum", "reddit"]
dataset_names = ["cnn_dailymail", "xsum", "reddit_tifu"]
highlights = [True, False, False]
make_splits = [False, False, True]
data_versions = ["3.0.0", "", "long"]
text_keys = ["article", "document", "documents"]
summary_keys = ["highlights", "summary", "tldr"]

idx = dataset_keys.index(args.dataset)
args.dataset_name = dataset_names[idx]
args.highlights = highlights[idx]
args.make_split = make_splits[idx]
args.data_version = data_versions[idx]
args.text_key = text_keys[idx]
args.summary_key = summary_keys[idx]

print("*"*50)
print(args)

sets = [("validation", "val"), ("test", "test"), ("train", "train")]
contents = [(args.text_key, "text"), (args.summary_key, "summary")]



def main(args):
    seed_everything(args.seed)

    if not(os.path.isdir("../../data/")):
        os.makedirs("../../data/")
    if not(os.path.isdir("../../data/{}/".format(args.dataset))):
        os.makedirs("../../data/{}/".format(args.dataset))

    if args.dataset in ["xsum"]:
        dataset = datasets.load_dataset(args.dataset_name)
    else:
        print(args.dataset_name, args.data_version)
        dataset = datasets.load_dataset(args.dataset_name, args.data_version)

    if args.make_split:
        dataset = dataset["train"]

        texts = [x[args.text_key] for x in dataset]
        summaries = [x[args.summary_key] for x in dataset]

        idx = np.random.permutation(len(texts))
        texts = [texts[i] for i in idx]
        summaries = [summaries[i] for i in idx]

        print(len(texts), len(summaries))
        print(texts[0])
        print("*" * 50)
        print(summaries[0])

        thresh = int(0.1 * len(texts))
        train_texts = texts[:(8 * thresh)]
        train_summaries = summaries[:(8 * thresh)]
        val_texts = texts[(8 * thresh):(9 * thresh)]
        val_summaries = summaries[(8 * thresh):(9 * thresh)]
        test_texts = texts[(9 * thresh):]
        test_summaries = summaries[(9 * thresh):]
        print(len(train_texts), len(val_texts), len(test_texts))

        set_texts = [train_texts, val_texts, test_texts]
        set_summaries = [train_summaries, val_summaries, test_summaries]
        set_names = ["train", "val", "test"]
        idx = 0
        for set_name in set_names:
            text = set_texts[idx]
            text_path = "../../data/{}/".format(args.dataset) + "/{}_text.txt".format(set_name)
            write_to_txt(text, text_path)

            summary = set_summaries[idx]
            summary_path = "../../data/{}/".format(args.dataset) + "/{}_summary.txt".format(set_name)
            write_to_txt(summary, summary_path)

            print(set_name, len(text), len(summary))
            idx += 1
    else:
        for x in sets:
            (set, set_name) = x
            for y in contents:
                (content, content_name) = y
                print(set, set_name, content, content_name)
                dataset_set = dataset[set]

                # single file with 1 data point per line
                text = [x[content].replace("\n", " ") for x in dataset_set]
                print(set, len(text))
                path = "../../data/{}/".format(args.dataset) + "/{}_{}.txt".format(set_name, content_name)
                write_to_txt(text, path)

                # individual files (to use for CNN/DM)
                if args.highlights:
                    text = [x[content] for x in dataset_set]
                    print(set, len(text))
                    folder_path = "../../data/{}/".format(args.dataset) + "/" + set_name + "/" + content_name + "/"
                    try:
                        os.makedirs(folder_path)
                    except FileExistsError:
                        pass
                    write_to_individual_txt(text, folder_path, set_name, content_name)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def write_to_txt(l, path):
    with open(path, "w") as f:
        for i in tqdm(range(len(l))):
            text = l[i]
            text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
            if len(text) == 0:
                print(i)
            f.write(text + "\n")


def write_to_individual_txt(l, path, set, content_name):
    for i in tqdm(range(len(l))):
        path_i = path + "{}_{}_{}.txt".format(set, content_name, i)
        with open(path_i, "w") as f:
            f.write(l[i])
            

if __name__ == '__main__':

    main(args)
