from __future__ import print_function, unicode_literals, division

import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from rouge_score import rouge_scorer
from bert_score import score
from common.bart_score import BARTScorer
from scipy.stats import pearsonr
from common.summary_processing import pre_rouge_processing



def overall_eval(val_texts, val_summaries, val_labels, args):
    # ROUGE
    all_score_names = []
    all_scores = []
    if args.eval_rouge:
        r1, r2, rl = rouge_eval("true labels", val_texts, val_summaries, val_labels, args)
        all_scores.append(r1)
        all_scores.append(r2)
        all_scores.append(rl)
        all_score_names += ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    # BERTScore
    if args.eval_bertscore:
        bs = bertscore_eval(val_summaries, val_labels, args)
        all_scores.append(bs)
        all_score_names.append("BERTScore")
    # BARTScore
    if args.eval_bartscore:
        bas = bartscore_eval(val_summaries, val_labels, args)
        all_scores.append(bas)
        all_score_names.append("BARTScore")
    # Abstractiveness
    if args.eval_new_ngram:
        new_ngram_eval(val_texts, val_summaries, args)
    # Overlap with source
    if args.eval_rouge_source:
        r1_text, r2_text, rl_text = rouge_eval("source", val_summaries, val_texts, val_texts, args)
        if args.check_correlation:
            r1_p = pearsonr(r1_true, r1_text)[0]
            r2_p = pearsonr(r2_true, r2_text)[0]
            rl_p = pearsonr(rl_true, rl_text)[0]
            print("Pearson correlations between ROUGE w true labels and ROUGE w source: {:.4f} / {:.4f} / {:.4f}".format(r1_p, r2_p, rl_p))

    return all_scores, all_score_names


def rouge_eval(mode, val_texts, val_summaries, val_labels, args):
    print("\n", "*"*10, "1 - ROUGE evaluation with {}".format(mode), "*"*10)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer = args.stemmer)
    all_r1s = []
    all_r2s = []
    all_rls = []
    for i in range(len(val_summaries)):
        summary = val_summaries[i]
        summary = pre_rouge_processing(summary, args)
        label = val_labels[i]
        r1, r2, rl = get_rouge_scores(summary, label, scorer, args)
        all_r1s.append(r1)
        all_r2s.append(r2)
        all_rls.append(rl)
    all_r1s = 100 * np.array(all_r1s)
    all_r2s = 100 * np.array(all_r2s)
    all_rls = 100 * np.array(all_rls)
    mean_r1 = np.mean(all_r1s)
    mean_r2 = np.mean(all_r2s)
    mean_rl = np.mean(all_rls)
    mean_r = (mean_r1 + mean_r2 + mean_rl) / 3
    print("Mean R: {:.4f}, R-1: {:.4f} (var: {:.4f}), R-2: {:.4f} (var: {:.4f}), R-L: {:.4f} (var: {:.4f})".format(
        mean_r, mean_r1, np.std(all_r1s), mean_r2, np.std(all_r2s), mean_rl, np.std(all_rls)))

    return all_r1s, all_r2s, all_rls


def get_rouge_scores(summary, label, scorer, args):
    rouge_scores = scorer.score(label, summary)
    r1 = rouge_scores["rouge1"].fmeasure
    r2 = rouge_scores["rouge2"].fmeasure
    rl = rouge_scores["rougeLsum"].fmeasure

    return r1, r2, rl


def bertscore_eval(val_summaries, val_labels, args, verbose=True):
    print("\n", "*" * 10, "2 - BERTScore evaluation", "*" * 10)
    p, r, f1 = score(val_summaries, val_labels, lang='en', verbose=verbose)
    mean_f1 = 100 * f1.mean()
    print("Mean BERTScore F1: {:.2f}".format(mean_f1))
    return 100 * f1.numpy()


def bartscore_eval(val_summaries, val_labels, args):
    print("\n", "*" * 10, "3 - BARTScore evaluation", "*" * 10)
    bart_scorer = BARTScorer(device = args.device, checkpoint = 'facebook/bart-large-cnn')
    bartscore_scores = bart_scorer.score(val_labels, val_summaries)
    m_bartscore = np.mean(np.array(bartscore_scores))
    print("Mean BARTScore: {:.2f}".format(m_bartscore))
    return np.array(bartscore_scores)


def new_ngram_eval(val_texts, val_summaries, args):
    print("\n", "*"*10, "5 - Abstractiveness / New n-gram", "*"*10)
    new_unigrams, new_bigrams, new_trigrams, new_quadrigrams = [], [], [], []
    for i in range(len(val_summaries)):
        # text
        text = val_texts[i].lower()
        text_words = word_tokenize(text)
        text_bigrams = [[text_words[j], text_words[j + 1]] for j in range(len(text_words) - 1)]
        text_trigrams = [[text_words[j], text_words[j + 1], text_words[j + 2]] for j in range(len(text_words) - 2)]
        text_quadrigrams = [[text_words[j], text_words[j + 1], text_words[j + 2], text_words[j + 3]] for j in range(len(text_words) - 3)]
        
        # summary
        summary = val_summaries[i].lower().replace("<n>", " ")
        summary_words = word_tokenize(summary)

        unigrams, bigrams, trigrams, quadrigrams = 0, 0, 0, 0
        for j in range(len(summary_words)):
            if not(summary_words[j] in text_words):
                unigrams += 1
            if j < len(summary_words) - 1:
                bigram = [summary_words[j], summary_words[j + 1]]
                if not(bigram in text_bigrams):
                    bigrams += 1
            if j < len(summary_words) - 2:
                trigram = [summary_words[j], summary_words[j + 1], summary_words[j + 2]]
                if not(trigram in text_trigrams):
                    trigrams += 1
            if j < len(summary_words) - 3:
                quadrigram = [summary_words[j], summary_words[j + 1], summary_words[j + 2], summary_words[j + 3]]
                if not(quadrigram in text_quadrigrams):
                    quadrigrams += 1
        if len(summary_words) > 0:
            new_unigrams.append(unigrams / (len(summary_words) - 0))
        if len(summary_words) > 1:
            new_bigrams.append(bigrams / (len(summary_words) - 1))
        if len(summary_words) > 2:
            new_trigrams.append(trigrams / (len(summary_words) - 2))
        if len(summary_words) > 3:
            new_quadrigrams.append(quadrigrams / (len(summary_words) - 3))
    new_unigrams = np.array(new_unigrams)
    m_uni = 100 * np.mean(new_unigrams)
    new_bigrams = np.array(new_bigrams)
    m_bi = 100 * np.mean(new_bigrams)
    new_trigrams = np.array(new_trigrams)
    m_tri = 100 * np.mean(new_trigrams)
    new_quadrigrams = np.array(new_quadrigrams)
    m_quadri = 100 * np.mean(new_quadrigrams)
    print("New unigrams: {:.2f}, bigrams: {:.2f}, trigrams: {:.2f}, quadrigrams: {:.2f}".format(m_uni, m_bi, m_tri, m_quadri))


