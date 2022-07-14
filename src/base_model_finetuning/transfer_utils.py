import torch

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer, T5ForConditionalGeneration,\
    PegasusTokenizer, PegasusModel, PegasusForConditionalGeneration, \
    BartTokenizer, BartForConditionalGeneration



def build_tokenizer(args):
    tokenizer = None
    if args.model_type.startswith("bart"):
        print("\nUsing Bart tokenizer")
        tokenizer = BartTokenizer.from_pretrained(args.model, cache_dir = args.cache_dir)
    elif args.model_type.startswith("pegasus"):
        print("\nUsing Pegasus tokenizer")
        tokenizer = PegasusTokenizer.from_pretrained(args.model, cache_dir = args.cache_dir)

    return tokenizer


def build_model(args):
    model = None
    if args.model_type.startswith("bart"):
        print("\nUsing Bart model")
        model = BartForConditionalGeneration.from_pretrained(args.model, cache_dir = args.cache_dir)
    elif args.model_type.startswith("pegasus"):
        print("\nUsing Pegasus model")
        model = PegasusForConditionalGeneration.from_pretrained(args.model, cache_dir = args.cache_dir)

    return model


def build_optimizer(model, args):
    optimizer = None
    if args.optimizer == "adam":
        print("\nUsing Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.optimizer == "adamw":
        print("\nUsing AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    return optimizer


def build_scheduler(optimizer, train_steps, args):
    scheduler = None
    if args.scheduler == "linear_warmup":
        print("\nUsing linear warmup scheduler")
        warmup_steps = int(args.warmup_ratio * train_steps)
        print("Number of warmup steps: {}".format(warmup_steps))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, train_steps)

    return scheduler


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)

    return tensors.detach()
