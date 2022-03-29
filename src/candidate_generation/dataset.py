from common.language_mapping import *



class Dataset:

    def __init__(self, mode, tokenizer, texts, summaries, args):
        self.mode = mode
        self.tokenizer = tokenizer
        self.texts = texts
        self.summaries = summaries
        self.args = args

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        if self.args.truncate_text:
            text = text[:self.args.max_text_size]
            #text = " ".join(text.split()[:400])
        summary = self.summaries[item]
        
        # add the prompt
        prompt = self.args.prompt
        if self.args.add_prompt_to_text:
            text = prompt + text

        text_inputs = self.tokenizer(text, return_tensors="pt", max_length=self.args.max_length, padding='max_length')
        text_inputs["input_ids"] = text_inputs["input_ids"][:, :self.args.max_length]
        text_inputs["attention_mask"] = text_inputs["attention_mask"][:, :self.args.max_length]
        
        summary_inputs = self.tokenizer(summary, return_tensors="pt", max_length=self.args.max_summary_length, padding='max_length')
        summary_inputs["input_ids"] = summary_inputs["input_ids"][:, :self.args.max_summary_length]
        summary_inputs["attention_mask"] = summary_inputs["attention_mask"][:, :self.args.max_summary_length]

        batch = {
            "text": text,
            "text_inputs": text_inputs,
            "summary": summary,
            "summary_inputs": summary_inputs,
        }

        return batch

