import random
import os
import numpy as np
import torch

from rouge_score import rouge_scorer



def seed_everything(seed=42):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def check_data_pipe(loaders):

    for loader in loaders:
        for idx, batch in enumerate(loader):
            print("*"*50)
            print(batch['text_lang'])
            print(batch['text_inputs']["input_ids"][:,:10])
            print(batch['summary_lang'])
            print(batch['summary_inputs']["input_ids"][:,:10])
            break


def display_losses(mode, losses):

    best_loss = np.min(np.array(losses))
    best_loss_idx = np.argmin(np.array(losses)) + 1
    print("Current {} loss is {:.4f}, best {} loss is {:.4f} achieved at iter {} / {}".format(mode, losses[-1], mode, best_loss, best_loss_idx, len(losses)))


def display_scores(mode, scores):

    for k in scores.keys():
        scores_k = scores[k]
        if "loss" in k:
            best_score_k = np.min(np.array(scores_k))
            best_score_k_idx = np.argmin(np.array(scores_k)) + 1
        else:
            best_score_k = np.max(np.array(scores_k))
            best_score_k_idx = np.argmax(np.array(scores_k)) + 1
        print("Best {} {} is {:.4f} achieved at iter {} / {}".format(mode, k, best_score_k, best_score_k_idx, len(scores_k)))


def compute_r1s(sents):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)

    all_r1s = []
    for i in range(len(sents)):
        pruned_sents = sents[:i] + sents[(i + 1):]
        pruned_text = " ".join(pruned_sents)
        scores = scorer.score(pruned_text, sents[i])
        r1 = 100 * scores["rouge1"].fmeasure
        all_r1s.append(r1)
    all_r1s = np.array(all_r1s)

    return all_r1s