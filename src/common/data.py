import os

from tqdm import tqdm



def load_data(set, args, individual_txt=False):
    if individual_txt:
        texts, summaries = read_data_files_individual(set, args)
    else:
        text_files, summary_files = prepare_data_files(set, args)
        texts, summaries = read_data_files(set, text_files, summary_files, args)

    print("Total # of texts: {}".format(len(texts)))

    return texts, summaries


def read_data_files_individual(set, args):
    texts = []
    summaries = []
    for lang in ["en"]:
        lang_path = args.data_folder + "{}/".format(lang)
        lang_set_text_path = lang_path + set + "/" + "text/"
        lang_set_summary_path = lang_path + set + "/" + "summary/"
        n_docs = len(os.listdir(lang_set_text_path))
        print("For lang {}, there are {} {} documents".format(lang, n_docs, set))
        for i in tqdm(range(n_docs)):
            text_path_i = lang_set_text_path + "{}_text_{}.txt".format(set, i)
            text_i = "".join(open(text_path_i, "r").readlines())
            texts.append(text_i)
        for i in tqdm(range(n_docs)):
            summary_path_i = lang_set_summary_path + "{}_summary_{}.txt".format(set, i)
            summary_i = "".join(open(summary_path_i, "r").readlines())
            summaries.append(summary_i)
    return texts, summaries


def prepare_data_files(set, args):
    # find the files
    text_files = []
    summary_files = []
    text_file = args.data_folder + "en/{}_text.txt".format(set)
    summary_file = args.data_folder  + "en/{}_summary.txt".format(set)
    text_files.append(text_file)
    summary_files.append(summary_file)

    print("For set {}, loading the following files:".format(set))
    print(text_files)
    print(summary_files)

    return text_files, summary_files


def read_data_files(set, text_files, summary_files, args):
    # read the .txt files
    texts = []
    summaries = []

    for text_file in text_files:
        lines = read_one_file(set, text_file, args)
        texts += lines
    for summary_file in summary_files:
        lines = read_one_file(set, summary_file, args)
        summaries += lines

    return texts, summaries


def read_one_file(set, file, args):
    lines = []
    with open(file, 'r') as f:
        for l in tqdm(f.readlines()):
            lines.append(l)
    lines = lines[:args.val_dataset_size]
    print(file, len(lines))

    return lines
