import os

from tqdm import tqdm



def load_data(set, args, individual_txt = False):
    if individual_txt:
        texts, summaries = read_data_files_individual(set, args)
    else:
        text_files, summary_files = prepare_data_files(set, args)
        texts, summaries = read_data_files(set, text_files, summary_files, args)

    print("Total # of texts: {}, # summaries: {}".format(len(texts), len(summaries)))

    print("\n")
    print(texts[0])
    print(summaries[0])

    return texts, summaries


def read_data_files_individual(set, args):
    texts = []
    summaries = []
    set_text_path = args.data_folder + "/" + set + "/" + "text/"
    set_summary_path = args.data_folder + "/" + set + "/" + "summary/"
    n_docs = len(os.listdir(set_text_path))
    print("There are {} {} documents".format(n_docs, set))
    for i in range(n_docs):
        text_path_i = set_text_path + "{}_text_{}.txt".format(set, i)
        text_i = "".join(open(text_path_i, "r").readlines())
        texts.append(text_i)
    for i in range(n_docs):
        summary_path_i = set_summary_path + "{}_summary_{}.txt".format(set, i)
        summary_i = "".join(open(summary_path_i, "r").readlines())
        summaries.append(summary_i)

    return texts, summaries


def prepare_data_files(set, args):
    # find the files
    text_files = []
    summary_files = []
    text_file = args.data_folder + "/{}_text.txt".format(set)
    text_files.append(text_file)
    summary_file = args.data_folder + "/{}_summary.txt".format(set)
    summary_files.append(summary_file)
    print(text_files)
    print(summary_files)

    return text_files, summary_files


def read_data_files(set, text_files, summary_files, args):
    # read the .txt files
    texts = []
    summaries = []
    idx = 0
    for text_file in text_files:
        with open(text_file, 'r') as f:
            lines = []
            for l in tqdm(f.readlines()):
                lines.append(l)
            print("# lines: {}".format(len(lines)))
            texts += lines

    for summary_file in summary_files:
        with open(summary_file, 'r') as f:
            lines = []
            for l in tqdm(f.readlines()):
                lines.append(l)
            print("# lines: {}".format(len(lines)))
            summaries += lines

    return texts, summaries


