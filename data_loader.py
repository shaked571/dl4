import os
import pickle
import zipfile

import torchtext.legacy as legacy
from vocab import Vocab

CACHE_DIR = '.cache'
TRAIN_CACHE_PATH = os.path.join(CACHE_DIR, 'train.pkl')
DEV_CACHE_PATH = os.path.join(CACHE_DIR, 'dev.pkl')
TEST_CACHE_PATH = os.path.join(CACHE_DIR, 'test.pkl')
VOCAB_CACHE_PATH = os.path.join(CACHE_DIR, 'vocab.pkl')

CACHE_FILES = [TRAIN_CACHE_PATH, DEV_CACHE_PATH, TEST_CACHE_PATH, VOCAB_CACHE_PATH]
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.data', "snli"))


def from_cache():
    files = []
    for fp in CACHE_FILES:
        with open(fp, 'rb') as f:
            cur_f = pickle.load(f)
            if fp == ANSWERS_CACHE_PATH:
                cur_f = remove_unk(cur_f)
            files.append(cur_f)
    return tuple(files)


def exists_cache():
    return all([os.path.isfile(f) for f in CACHE_FILES])


def save_pkl(f2dump, out_path):
    with open(out_path, 'wb') as handle:
        pickle.dump(f2dump, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_cache(train, dev, test, vocab):
    if not os.path.isdir(CACHE_DIR):
        os.mkdir(CACHE_DIR)
    save_pkl(train, TRAIN_CACHE_PATH)
    save_pkl(dev, DEV_CACHE_PATH)
    save_pkl(test, TEST_CACHE_PATH)
    save_pkl(vocab, VOCAB_CACHE_PATH)


def load_snli():
    if exists_cache():
        return from_cache()
    else:
        inputs = legacy.data.Field(
            tokenize='spacy',
            tokenizer_language='en_core_web_sm',
            lower=True,
        )
        target = legacy.data.Field(sequential=False, is_target=True)

        # make splits for data
        try:
            train, dev, test = legacy.datasets.SNLI.splits(
                text_field=inputs, label_field=target
            )
        except OSError:
            with zipfile.ZipFile(".data/snli/snli_1.0.zip", "r") as zip_ref:
                members2extract = ['snli_1.0/',  'snli_1.0/snli_1.0_dev.jsonl',
                                   'snli_1.0/snli_1.0_dev.txt', 'snli_1.0/snli_1.0_test.jsonl',
                                   'snli_1.0/snli_1.0_test.txt', 'snli_1.0/snli_1.0_train.jsonl',
                                   'snli_1.0/snli_1.0_train.txt']
                zip_ref.extractall(DATA_PATH, members2extract)
            train, dev, test = legacy.datasets.SNLI.splits(
                text_field=inputs, label_field=target
            )

        inputs.build_vocab(train, min_freq=1, vectors="glove.6B.300d")
        target.build_vocab(train)
        vocab = Vocab(inputs, taret)
        save_cache(list(train), list(dev), list(test), vocab)
    return train, dev, test, vocab
