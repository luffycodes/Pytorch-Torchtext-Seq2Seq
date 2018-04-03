import torch
from torchtext import data
from torchtext import datasets
import time
import re
import spacy
import os
from tqdm import tqdm
import logging

SOS_WORD = '<SOS>'
EOS_WORD = '<EOS>'
PAD_WORD = '<PAD>'


def get_tokenizer(lang):
    spacy_lang = spacy.load(lang)
    return lambda s: [tok.text for tok in spacy_lang.tokenizer(s)]


def generate_fields(src_lang, trg_lang):
    src_field = data.Field(tokenize=get_tokenizer(src_lang),
                           init_token=SOS_WORD,
                           eos_token=EOS_WORD,
                           pad_token=PAD_WORD,
                           include_lengths=True,
                           batch_first=True)

    trg_field = data.Field(tokenize=get_tokenizer(trg_lang),
                           init_token=SOS_WORD,
                           eos_token=EOS_WORD,
                           pad_token=PAD_WORD,
                           include_lengths=True,
                           batch_first=True)

    return src_field, trg_field


def save_data(data_file, dataset):
    examples = vars(dataset)['examples']
    dataset = {'examples': examples}

    torch.save(dataset, data_file)


class MaxlenTranslationDataset(data.Dataset):
    # Code modified from
    # https://github.com/pytorch/text/blob/master/torchtext/datasets/translation.py
    # to be able to control the max length of the source and target sentences

    def __init__(self, path, exts, fields, max_len=None, **kwargs):

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in tqdm(zip(src_file, trg_file)):
                src_line, trg_line = src_line.split(' '), trg_line.split(' ')
                if max_len is not None:
                    src_line = src_line[:max_len]
                    src_line = str(' '.join(src_line))
                    trg_line = trg_line[:max_len]
                    trg_line = str(' '.join(trg_line))

                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(MaxlenTranslationDataset, self).__init__(examples, fields, **kwargs)


class DataPreprocessor(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.src_field = None
        self.trg_field = None
        self.console_logger = logging.getLogger()

    def getDatasets(self, args):
        src_lang = args.src_lang
        trg_lang = args.trg_lang
        max_len = args.max_len

        val_file = os.path.join(args.data_path,
                                "data_dev_{}_{}_{}.json".format(src_lang, trg_lang, max_len))

        val_dataset = self.getOneDataset(args.val_path, val_file, src_lang, trg_lang, max_len)

        train_file = os.path.join(args.data_path, "data_{}_{}_{}_{}.json".format(args.dataset, src_lang,
                                                                                 trg_lang, max_len))
        train_dataset = self.getOneDataset(args.train_path, train_file, src_lang, trg_lang, max_len)

        sts_file = os.path.join(args.data_path, "data_sts_{}_{}_{}.json".format(src_lang, trg_lang, max_len))
        sts_dataset = self.getOneDataset(args.sts_path, sts_file, args.sts_src_lang, args.sts_trg_lang, max_len)

        return train_dataset, val_dataset, sts_dataset

    def getOneDataset(self, dataset_path, dataset_processed_file, src_lang, trg_lang, max_len):
        if os.path.isfile(dataset_processed_file):
            self.console_logger.debug("loading preprocessed data from file path: %s", dataset_processed_file)
            return self.loadDataset(dataset_processed_file)
        else:
            self.console_logger.debug("reading data for first time from file path: %s", dataset_path)
            return self.preprocess(dataset_path, dataset_processed_file, src_lang, trg_lang, max_len)

    def preprocess(self, dataset_path, dataset_processed_file, src_lang, trg_lang, max_len=None):
        self.logger.debug("Preprocessing dataset from file path: %s", dataset_path)
        dataset = self.getMaxlenTranslationDataset(dataset_path, src_lang, trg_lang, max_len)

        self.logger.debug("Saving dataset to file: %s", dataset_processed_file)
        save_data(dataset_processed_file, dataset)

        return dataset

    def loadDataset(self, dataset_processed_file):
        # Loading saved data
        dataset_processed = torch.load(dataset_processed_file)
        dataset_examples = dataset_processed['examples']

        # Generating torchtext dataset class
        fields = [('src', self.src_field), ('trg', self.trg_field)]
        dataset_processed = data.Dataset(fields=fields, examples=dataset_examples)

        return dataset_processed

    def buildVocab(self, dataset1, dataset2):
        # Building field vocabulary
        self.src_field.build_vocab(dataset1, dataset2, min_freq=3, max_size=500000)
        self.trg_field.build_vocab(dataset1, min_freq=3, max_size=500000)

        src_vocab, trg_vocab, src_inv_vocab, trg_inv_vocab = self.generate_vocabs()
        vocabs = {'src_vocab': src_vocab, 'trg_vocab': trg_vocab,
                  'src_inv_vocab': src_inv_vocab, 'trg_inv_vocab': trg_inv_vocab}

        return vocabs

    def getMaxlenTranslationDataset(self, data_path, src_lang, trg_lang, max_len=None):
        exts = ('.' + src_lang, '.' + trg_lang)

        dataset = MaxlenTranslationDataset(
            path=data_path,
            exts=exts,
            fields=(self.src_field, self.trg_field),
            max_len=max_len)

        return dataset

    def generate_vocabs(self):
        # Define string to index vocabs
        src_vocab = self.src_field.vocab.stoi
        trg_vocab = self.trg_field.vocab.stoi

        # Define index to string vocabs
        src_inv_vocab = self.src_field.vocab.itos
        trg_inv_vocab = self.trg_field.vocab.itos

        return src_vocab, trg_vocab, src_inv_vocab, trg_inv_vocab
