import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
PREFIX = "translate English to SQL: "
MAX_INPUT_LEN = 128    # longest question is 65 tokens
MAX_TARGET_LEN = 512   # longest SQL query is 511 tokens

def find_unk_chars(tokenizer, texts):
    """Characters that the T5 vocabulary cannot represent (they would become <unk>)."""
    chars = set("".join(texts))
    return {c for c in chars
            if c.strip() and tokenizer.unk_token_id in tokenizer(c, add_special_tokens=False)["input_ids"]}

def protect(s, unk_chars):
    """Replace each unknown character with a placeholder T5 can tokenize, e.g. '<' -> 'UNKCHAR60'."""
    return "".join(f"UNKCHAR{ord(c)}" if c in unk_chars else c for c in s)

def restore_sql(s):
    """Map placeholders back to the original characters and normalize whitespace."""
    s = re.sub(r"UNKCHAR(\d+)", lambda m: chr(int(m.group(1))), s)
    return " ".join(s.split())

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Data processing for T5: adds a task prefix to each question, normalizes
        whitespace in the SQL, and tokenizes both with the google-t5/t5-small tokenizer.
        The decoder input starts with <extra_id_0> as the beginning-of-sentence token.
        On the test set there is no SQL, so only encoder inputs are prepared.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.bos_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl = load_lines(os.path.join(data_folder, f'{split}.nl'))
        nl_unk = find_unk_chars(tokenizer, nl)
        inputs = [PREFIX + protect(q, nl_unk) for q in nl]
        enc_ids = tokenizer(inputs, max_length=MAX_INPUT_LEN, truncation=True)['input_ids']

        if split == 'test':
            return [(torch.tensor(e), torch.tensor([self.bos_id])) for e in enc_ids]

        sql = load_lines(os.path.join(data_folder, f'{split}.sql'))
        sql = [" ".join(s.split()) for s in sql]
        sql_unk = find_unk_chars(tokenizer, sql)
        sql = [protect(s, sql_unk) for s in sql]
        tgt_ids = tokenizer(sql, max_length=MAX_TARGET_LEN, truncation=True)['input_ids']  # ends with </s>

        data = []
        for e, t in zip(enc_ids, tgt_ids):
            dec_input = [self.bos_id] + t[:-1]   # shifted right, starts with BOS
            data.append((torch.tensor(e), torch.tensor(dec_input), torch.tensor(t)))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Dynamic padding for training and dev evaluation.
    Returns encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs.
    '''
    encoder_ids = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence([b[2] for b in batch], batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack([b[1][:1] for b in batch])   # B x 1, the BOS token
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Dynamic padding for inference on the test set.
    Returns encoder_ids, encoder_mask, initial_decoder_inputs.
    '''
    encoder_ids = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack([b[1] for b in batch])        # B x 1, the BOS token
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
