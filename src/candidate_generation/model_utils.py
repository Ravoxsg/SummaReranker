
from transformers import T5Tokenizer, T5ForConditionalGeneration, \
    PegasusTokenizer, PegasusForConditionalGeneration, \
    BartTokenizerFast, BartForConditionalGeneration



def build_tokenizer(args):
    tokenizer = None
    if args.model_type.startswith("t5"):
        print("\nUsing T5 tokenizer")
        tokenizer = T5Tokenizer.from_pretrained(args.model, cache_dir = args.cache_dir)
    elif args.model_type.startswith("pegasus"):
        print("\nUsing Pegasus tokenizer")
        tokenizer = PegasusTokenizer.from_pretrained(args.model, cache_dir = args.cache_dir)
    elif args.model_type.startswith("bart"):
        print("\nUsing Bart tokenizer")
        tokenizer = BartTokenizerFast.from_pretrained(args.model, cache_dir = args.cache_dir)

    return tokenizer


def build_model(args):
    model = None
    if args.model_type.startswith("t5"):
        print("\nUsing T5 model")
        model = T5ForConditionalGeneration.from_pretrained(args.model, cache_dir = args.cache_dir)
    elif args.model_type.startswith("pegasus"):
        print("\nUsing Pegasus model")
        model = PegasusForConditionalGeneration.from_pretrained(args.model, cache_dir = args.cache_dir)
    elif args.model_type.startswith("bart"):
        print("\nUsing Bart model")
        model = BartForConditionalGeneration.from_pretrained(args.model, cache_dir = args.cache_dir)

    return model
