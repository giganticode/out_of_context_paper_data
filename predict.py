from transformers import T5ForConditionalGeneration, AutoTokenizer
import pandas as pd
import numpy as np
from pathlib import Path
import fire
import re
from pprint import pprint
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
from transformers import DataCollatorForSeq2Seq

from functools import partial

DEVICE = 'cuda'

class BugDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.tokenizer = tokenizer
        self.df = df
        self.old_values = df.old.values
        self.new_values = df.new.values
        self.len = self.df.shape[0]

    def process_sample(self, index):
        source = self.old_values[index]
        target = self.new_values[index]
        self.tokenizer.src_lang = 'java'
        self.tokenizer.tgt_lang = 'java'
        model_inputs = self.tokenizer(source, max_length=1024, padding=False, truncation=True)
        return model_inputs

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.process_sample(i)


def main(model_name_or_path, input_filename, output_filename, min_hunk_count, max_hunk_count, nrows=None, early_stopping=False, batch_size=32, n=5, max_length=1024):
    print(input_filename, batch_size, n, max_length, early_stopping)

    if os.path.exists(output_filename):
        raise ValueError(f"output file '{output_filename}' already exists")

    df = pd.read_json(input_filename, orient='records', lines=True, nrows=nrows, engine='pyarrow')
    if 'train' not in df:
        print("Assuming all samples are training samples!!!!")
        df['train'] = False
    df = df[~df.train & (df.hunk_count <= max_hunk_count) & (df.hunk_count >= min_hunk_count) & (df.context_line_count > 0)]

    print(df.shape)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    model = model.to(DEVICE)
    model.eval()

    dataset = BugDataset(df, tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=12, shuffle=False, collate_fn=data_collator)

    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask = batch['input_ids'], batch['attention_mask'] # , ids = batch
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            print(input_ids.shape)
            preds = model.generate(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   length_penalty=0.0,
                                   max_length=max_length,
                                   num_beams=n,
                                   do_sample=False,
                                   early_stopping=early_stopping,
                                   num_return_sequences=n)
                                   #penalty_alpha=0.6, top_k=4)


            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_preds = [decoded_preds[i:i+n] for i in range(0, len(decoded_preds), n)]

            all_preds.extend(decoded_preds)

    rows = []
    assert len(all_preds) == df.shape[0]
    for i, preds in enumerate(all_preds):
        input_row = df.iloc[i]
        output_row = {'change_count': input_row.change_count, 'hunk_count': input_row.hunk_count, 'preds': preds}
        for attr in ['filename', 'repo_name', 'sha', 'label', 'labels']:
            if attr in df:
                output_row[attr] = input_row[attr]
        rows.append(output_row)

    df = pd.DataFrame(rows)
    df.to_json(output_filename, lines=True, orient='records')

if __name__ == '__main__':
    fire.Fire(main)