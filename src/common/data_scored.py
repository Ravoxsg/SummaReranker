import pickle
import os

from tqdm import tqdm



def load_data(set, size, args, individual_txt=False, train=False):
    # texts & summaries
    if individual_txt:
        texts, summaries = read_data_files_individual(set, args, train=train)
    else:
        text_files, summary_files = prepare_data_files(set, args, train=train)
        texts, summaries = read_data_files(text_files, summary_files, args)

    # scored summaries - with multiple scores
    if train:
        all_scored_summaries = []
        for generation_method in args.generation_methods:
            gen_scored_summaries = []
            for j in range(len(args.scoring_methods)):
                scored_summaries_j = []
                for i in range(len(set)):
                    set_ = set[i]
                    size_ = size[i]
                    model_name_ = args.train_model_names[i]
                    print(set_)
                    print(size_)
                    print(model_name_)
                    scored_summaries_path_j_i = "../../scored_summaries/{}/{}/{}/{}/{}_scored_summaries_{}_{}_beams_{}.pkl".format(
                        args.dataset, set_, generation_method, args.scoring_methods[j],
                        set_, model_name_, size_, args.num_beams
                    )
                    print(scored_summaries_path_j_i)
                    with open(scored_summaries_path_j_i, "rb") as f:
                        scored_summaries_j_i = pickle.load(f)
                    scored_summaries_j += scored_summaries_j_i
                gen_scored_summaries.append(scored_summaries_j)
            scored_summaries = []
            for i in range(len(gen_scored_summaries[0])):
                summaries_i = gen_scored_summaries[0][i][0]
                scores_i = []
                for j in range(len(args.scoring_methods)):
                    scores_i_j = gen_scored_summaries[j][i][1]
                    scores_i.append(scores_i_j)
                scored_summaries.append((summaries_i, scores_i))
            print(len(scored_summaries), len(scored_summaries[0]), len(scored_summaries[0][0]), len(scored_summaries[0][1]), len(scored_summaries[0][1][0]))
            all_scored_summaries.append(scored_summaries)
        scored_summaries = combine_summaries(all_scored_summaries)
    else:
        all_scored_summaries = []
        for generation_method in args.generation_methods:
            gen_scored_summaries = []
            for j in range(len(args.scoring_methods)):
                scored_summaries_path_j = "../../scored_summaries/{}/{}/{}/{}/{}_scored_summaries_{}_{}_beams_{}.pkl".format(
                    args.dataset, set, generation_method, args.scoring_methods[j],
                    set, args.model_name, size, args.num_beams
                )
                print(scored_summaries_path_j)
                with open(scored_summaries_path_j, "rb") as f:
                    scored_summaries = pickle.load(f)
                gen_scored_summaries.append(scored_summaries)
            scored_summaries = []
            for i in range(len(gen_scored_summaries[0])):
                summaries_i = gen_scored_summaries[0][i][0]
                scores_i = []
                for j in range(len(args.scoring_methods)):
                    scores_i_j = gen_scored_summaries[j][i][1]
                    scores_i.append(scores_i_j)
                scored_summaries.append((summaries_i, scores_i))
            print(len(scored_summaries), len(scored_summaries[0]), len(scored_summaries[0][0]), len(scored_summaries[0][1]), len(scored_summaries[0][1][0]))
            all_scored_summaries.append(scored_summaries)
        scored_summaries = combine_summaries(all_scored_summaries)

    print("Total # of texts: {}, labels: {}, summary_candidates: {}, # candidates / text: {}".format(
        len(texts), len(summaries), len(scored_summaries), len(scored_summaries[0][0])))

    return texts, summaries, scored_summaries


def read_data_files_individual(set, args, train=False):
    texts = []
    summaries = []
    if train:
        for set_ in set:
            set_text_path = "../../data/{}/{}_text.txt".format(args.dataset, set_)
            set_summary_path = "../../data/{}/{}_summary.txt".format(args.dataset, set_)
            n_docs = len(os.listdir(set_text_path))
            print("There are {} {} documents".format(n_docs, set_))
            for i in tqdm(range(n_docs)):
                text_path_i = set_text_path + "{}_text_{}.txt".format(set_, i)
                text_i = "".join(open(text_path_i, "r").readlines())
                texts.append(text_i)
            for i in tqdm(range(n_docs)):
                summary_path_i = set_summary_path + "{}_summary_{}.txt".format(set_, i)
                summary_i = "".join(open(summary_path_i, "r").readlines())
                summaries.append(summary_i)
    else:
        set_text_path = "../../data/{}/{}_text.txt".format(args.dataset, set)
        set_summary_path = "../../data/{}/{}_summary.txt".format(args.dataset, set)
        n_docs = len(os.listdir(set_text_path))
        print("There are {} {} documents".format(n_docs, set))
        for i in tqdm(range(n_docs)):
            text_path_i = set_text_path + "{}_text_{}.txt".format(set, i)
            text_i = "".join(open(text_path_i, "r").readlines())
            texts.append(text_i)
        for i in tqdm(range(n_docs)):
            summary_path_i = set_summary_path + "{}_summary_{}.txt".format(set, i)
            summary_i = "".join(open(summary_path_i, "r").readlines())
            summaries.append(summary_i)

    return texts, summaries


def prepare_data_files(set, args, train):
    text_files = []
    summary_files = []    
    if train:
        for set_ in set:
            text_file = "../../data/{}/{}_text.txt".format(args.dataset, set_)
            summary_file = "../../data/{}/{}_summary.txt".format(args.dataset, set_)
            text_files.append(text_file)
            summary_files.append(summary_file)
    else:
        text_file = "../../data/{}/{}_text.txt".format(args.dataset, set)
        summary_file = "../../data/{}/{}_summary.txt".format(args.dataset, set)
        text_files.append(text_file)
        summary_files.append(summary_file)

    print("For set {}, loading the following files:".format(set))
    print(text_files)
    print(summary_files)

    return text_files, summary_files


def read_data_files(text_files, summary_files, args):
    # read the .txt files
    texts = []
    summaries = []

    for text_file in text_files:
        lines = read_one_file(text_file, args)
        texts += lines
    for summary_file in summary_files:
        lines = read_one_file(summary_file, args)
        summaries += lines

    return texts, summaries


def read_one_file(file, args):
    lines = []
    with open(file, 'r') as f:
        for l in tqdm(f.readlines()):
            lines.append(l)
    print(file, len(lines))

    return lines


def combine_summaries(all_scored_summaries):
    res = []
    for i in tqdm(range(len(all_scored_summaries[0]))):
        summaries_i = []
        scores_i = []
        for k in range(len(all_scored_summaries[0][i][1])):
            scores_i.append([])
        for j in range(len(all_scored_summaries)):
            summaries_i_j = all_scored_summaries[j][i][0]
            summaries_i += summaries_i_j
            scores_i_j = all_scored_summaries[j][i][1]
            for k in range(len(scores_i_j)):
                scores_i[k] += scores_i_j[k]
        res.append((summaries_i, scores_i))

    return res
