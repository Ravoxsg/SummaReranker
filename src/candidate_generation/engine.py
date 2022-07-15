import pickle
import torch
import gc 

from tqdm import tqdm

from transformers import BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, HammingDiversityLogitsProcessor



def get_summaries(tokenizer, val_loader, model, device, args):
    val_texts = []
    val_summaries = []
    val_labels = []
    base_model = model.pretrained_model

    for idx, batch in tqdm(enumerate(val_loader)):
        for k in batch["text_inputs"].keys():
            batch["text_inputs"][k] = batch["text_inputs"][k].to(device)
            if len(batch["text_inputs"][k].shape) > 2:
                batch["text_inputs"][k] = batch["text_inputs"][k].squeeze(1)

        model.zero_grad()
        val_texts += batch["text"]

        raw_summaries = beam_search_step(batch, tokenizer, base_model, device, args)
        
        summaries = []
        for i in range(len(batch["text"])):
            summaries.append(raw_summaries[i*args.num_return_sequences:(i+1)*args.num_return_sequences])
        val_summaries += summaries

        labels = batch["summary"]
        val_labels += labels

    print(len(val_texts), len(val_summaries), len(val_summaries[0]), len(val_labels))

    return val_texts, val_summaries, val_labels


def beam_search_step(batch, tokenizer, base_model, device, args):
    # 1 - beam search
    if args.generation_method == "beam_search":
        summary_ids = base_model.generate(
            batch["text_inputs"]['input_ids'],
            attention_mask = batch["text_inputs"]["attention_mask"],
            num_beams = args.num_beams,
            num_return_sequences = args.num_return_sequences,
            max_length = args.max_summary_length,
            repetition_penalty = args.repetition_penalty,
            length_penalty = args.length_penalty,
            no_repeat_ngram_size = args.no_repeat_ngram_size,
            use_cache = True,
            early_stopping = True
        )
    # 2 - diverse beam search
    if args.generation_method == "diverse_beam_search":
        summary_ids = base_model.generate(
            batch["text_inputs"]['input_ids'],
            attention_mask = batch["text_inputs"]["attention_mask"],
            num_beams = args.num_beams,
            num_beam_groups = args.num_beam_groups,
            num_return_sequences = args.num_return_sequences,
            max_length = args.max_summary_length,
            diversity_penalty = args.diversity_penalty,
            repetition_penalty = args.repetition_penalty,
            length_penalty = args.length_penalty,
            no_repeat_ngram_size = args.no_repeat_ngram_size,
            use_cache = True,
            early_stopping = True
        )
    # 3 - top-p sampling
    if args.generation_method == "top_p_sampling":
        summary_ids = base_model.generate(
            batch["text_inputs"]['input_ids'],
            attention_mask = batch["text_inputs"]["attention_mask"],
            num_beams = 1,
            do_sample = True,
            top_p = args.top_p,
            num_return_sequences = args.num_return_sequences,
            max_length = args.max_summary_length,
            repetition_penalty = args.repetition_penalty,
            length_penalty = args.length_penalty,
            no_repeat_ngram_size = args.no_repeat_ngram_size,
            use_cache = True,
            early_stopping = True
        )
    # 4 - top-k sampling
    if args.generation_method == "top_k_sampling":
        summary_ids = base_model.generate(
            batch["text_inputs"]['input_ids'],
            attention_mask = batch["text_inputs"]["attention_mask"],
            num_beams = 1,
            do_sample = True,
            top_k = args.top_k,
            num_return_sequences = args.num_return_sequences,
            max_length = args.max_summary_length,
            repetition_penalty = args.repetition_penalty,
            length_penalty = args.length_penalty,
            no_repeat_ngram_size = args.no_repeat_ngram_size,
            use_cache = True,
            early_stopping = True
        )
    generated = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    del summary_ids
    gc.collect()

    return generated
