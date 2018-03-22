from torchtext import data
from torchtext import datasets

import re
import spacy

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

url = re.compile('(<url>.*</url>)')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]


# Testing IWSLT
DE = data.Field(tokenize=tokenize_de)
EN = data.Field(tokenize=tokenize_en)

# https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8
train, val, test = datasets.WMT14.splits(exts=('.de', '.en'), fields=(DE, EN))

print(train.fields)
print(len(train))
print(vars(train[0]))
print(vars(train[100]))

DE.build_vocab(train.src, min_freq=3)
EN.build_vocab(train.trg, max_size=50000)

train_iter, val_iter = data.BucketIterator.splits(
    (train, val), batch_size=3, device=0)

print(DE.vocab.freqs.most_common(10))
print(len(DE.vocab))
print(EN.vocab.freqs.most_common(10))
print(len(EN.vocab))

batch = next(iter(train_iter))
print(batch.src)
print(batch.trg)