import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    raise NotImplementedError

    ##### YOUR CODE ENDS HERE ######

    return example


# ===================== Q2: custom transformation =====================
import random
import re

MODE = "casual"

TYPO_PROB = 0.2          # typo mode: chance a word (4+ letters) gets a typo
CASUAL_TYPO_PROB = 0.30
ABBREV_PROB = 0.8        # casual mode: chance each phrase/word is abbreviated
PERIOD_DROP_PROB = 0.5   # casual mode: chance a sentence-ending period is dropped

KEYBOARD = {
    'q': 'wa', 'w': 'qeas', 'e': 'wrsd', 'r': 'etdf', 't': 'ryfg', 'y': 'tugh',
    'u': 'yihj', 'i': 'uojk', 'o': 'ipkl', 'p': 'ol', 'a': 'qwsz', 's': 'awedxz',
    'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn',
    'k': 'jiolm', 'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb',
    'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
}
PHRASES = {
    "to be honest": "tbh", "i don't know": "idk", "i do not know": "idk",
    "oh my god": "omg", "by the way": "btw", "in my opinion": "imo",
    "not going to lie": "ngl", "for real": "fr", "right now": "rn",
    "at the moment": "atm", "going to": "gonna", "want to": "wanna",
    "got to": "gotta", "kind of": "kinda", "sort of": "sorta",
}
WORDS = {
    "you're": "ur", "you": "u", "your": "ur", "are": "r", "really": "rly",
    "because": "bc", "people": "ppl", "though": "tho", "although": "tho",
    "something": "smth", "probably": "prob", "before": "b4", "great": "gr8",
    "okay": "ok", "please": "pls", "thanks": "thx", "definitely": "def",
    "seriously": "srsly", "favorite": "fave", "favourite": "fave",
    "about": "abt", "tonight": "2nite", "with": "w/", "without": "w/o",
    "and": "&", "why": "y", "love": "luv", "everyone": "every1",
    "someone": "some1", "anyone": "any1", "tomorrow": "tmrw",
    "minutes": "mins", "nothing": "nothin", "going": "goin",
}

def _typo(w):
    i = random.randrange(len(w))
    kind = random.choice(["key", "swap", "drop"])
    c = w[i].lower()
    if kind == "key" and c in KEYBOARD:
        new = random.choice(KEYBOARD[c])
        new = new.upper() if w[i].isupper() else new
        return w[:i] + new + w[i + 1:]
    if kind == "swap" and i < len(w) - 1:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w[:i] + w[i + 1:]

def _add_typos(text, prob):
    return re.sub(r"[A-Za-z]{4,}",
                  lambda m: _typo(m.group(0)) if random.random() < prob else m.group(0),
                  text)

def _replace_all(text, table):
    for k in sorted(table, key=len, reverse=True):
        text = re.sub(r"\b" + re.escape(k) + r"\b",
                      lambda m: table[k] if random.random() < ABBREV_PROB else m.group(0),
                      text)
    return text

def custom_transform(example):
    text = example["text"]
    if MODE == "typo":
        text = _add_typos(text, TYPO_PROB)
    else:  # casual texting style
        text = text.lower()
        text = _replace_all(text, PHRASES)
        text = _replace_all(text, WORDS)
        text = re.sub(r"(\w)'(\w)", r"\1\2", text)
        text = _add_typos(text, CASUAL_TYPO_PROB)
        text = re.sub(r"\.(\s|$)",
                      lambda m: m.group(1) if random.random() < PERIOD_DROP_PROB else m.group(0),
                      text)
    example["text"] = text
    return example
