from nltk.tokenize import sent_tokenize



def pre_rouge_processing(summary, args):
    if args.clean_n:
        summary = summary.replace("<n>", " ")
    if args.highlights:
        summary = "\n".join(sent_tokenize(summary))
    return summary