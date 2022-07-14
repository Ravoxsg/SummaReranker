import gc
import numpy as np
import sys

sys.path.append("/data/mathieu/SummaReranker/src/")

from tqdm import tqdm

from common.evaluation import overall_eval



def validate(mode, loader, all_losses, tokenizer, model, device, args):
    from time import time

    print("\nStart evaluation on {}...".format(mode))
    ta = time()

    model.eval()

    # train
    val_losses = []
    val_texts = []
    val_summaries = []
    val_labels = []
    val_times = []
    for idx, batch in tqdm(enumerate(loader)):

        model.zero_grad()

        t1 = time()

        val_texts += batch["text"]
        if args.inference:
            loss = inference_step(batch, model, device)
            val_losses.append(loss.item())
        if args.generation:
            summaries = generation_step(batch, tokenizer, model, device, args)
            val_summaries += summaries
            labels = batch["summary"]
            val_labels += labels
        
        t2 = time()
        val_times.append(t2-t1)

        del batch
        gc.collect()

    if len(val_losses) > 0:
        m_loss = np.mean(np.array(val_losses))
        all_losses.append(m_loss)
        print("\nMean validation loss: {:.4f}".format(m_loss))

    if len(val_summaries) > 0:
        print(len(val_texts), len(val_summaries), len(val_labels))
        overall_eval(val_texts, val_summaries, val_labels, args)

    tb = time()
    print("finished evaluation in {:.4f} \n".format(tb - ta))

    return all_losses


def inference_step(batch, model, device):
    text_inputs = batch["text_inputs"]
    summary_inputs = batch["summary_inputs"]

    for k in text_inputs.keys():
        text_inputs[k] = text_inputs[k].squeeze(1).to(device)
        summary_inputs[k] = summary_inputs[k].squeeze(1).to(device)

    labels = summary_inputs["input_ids"]
    outputs = model(**text_inputs, labels=labels)
    loss = outputs["loss"]

    return loss


def generation_step(batch, tokenizer, model, device, args):
    model_to_use = model.pretrained_model
    for k in batch["text_inputs"].keys():
        batch["text_inputs"][k] = batch["text_inputs"][k].to(device)
        if len(batch["text_inputs"][k].shape) > 2:
            batch["text_inputs"][k] = batch["text_inputs"][k].squeeze(1)

    if args.model_type in ["t5", "mt5", "pegasus"]:
        summary_ids = model_to_use.generate(
            batch["text_inputs"]['input_ids'],
            attention_mask = batch["text_inputs"]["attention_mask"],
            use_cache = True,
            num_beams = args.num_beams,
            num_return_sequences = args.num_return_sequences,
            #min_length = args.min_summary_length,
            max_length = args.max_summary_length,
            early_stopping = True,
            repetition_penalty = args.repetition_penalty,
            length_penalty = args.length_penalty,
            no_repeat_ngram_size = args.no_repeat_ngram_size
        )
    elif args.model_type in ["bart", "mbart"]:
        summary_ids = model_to_use.generate(
            batch["text_inputs"]['input_ids'],
            num_beams = args.num_beams,
            min_length = args.min_summary_length,
            max_length = args.max_summary_length,
            early_stopping = True,
            repetition_penalty = args.repetition_penalty,
            length_penalty = args.length_penalty,
            no_repeat_ngram_size = args.no_repeat_ngram_size
        )
    generated = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return generated



