import os
import pickle
import zipfile

import torchtext.legacy as legacy

UNIQUE = '<uuukkk>'
CACHE_DIR = '.cache'
TRAIN_CACHE_PATH = os.path.join(CACHE_DIR, 'train.pkl')
DEV_CACHE_PATH = os.path.join(CACHE_DIR, 'dev.pkl')
TEST_CACHE_PATH = os.path.join(CACHE_DIR, 'test.pkl')
INPUT_CACHE_PATH = os.path.join(CACHE_DIR, 'input.pkl')
ANSWERS_CACHE_PATH = os.path.join(CACHE_DIR, 'answers.pkl')

CACHE_FILES = [TRAIN_CACHE_PATH, DEV_CACHE_PATH, TEST_CACHE_PATH, INPUT_CACHE_PATH, ANSWERS_CACHE_PATH]
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.data', "snli"))


def from_cache():
    files = []
    for fp in CACHE_FILES:
        with open(fp, 'rb') as f:
            cur_f = pickle.load(f)
            files.append(cur_f)
    return tuple(files)

def exists_cache():
    return all([os.path.isfile(f) for f in CACHE_FILES])

def save_pkl(f2dump, out_path):
    with open(out_path, 'wb') as handle:
        pickle.dump(f2dump, handle, protocol=pickle.HIGHEST_PROTOCOL)



def save_cache(train, dev, test, inputs, answers):
    if not os.path.isdir(CACHE_DIR):
        os.mkdir(CACHE_DIR)
    save_pkl(train, TRAIN_CACHE_PATH)
    save_pkl(dev, DEV_CACHE_PATH)
    save_pkl(test, TEST_CACHE_PATH)
    save_pkl(inputs, INPUT_CACHE_PATH)
    save_pkl(answers, ANSWERS_CACHE_PATH)



def load_snli():
    if exists_cache():
        return from_cache()
    else:
        inputs = legacy.data.Field(
            init_token="</s>",
            tokenize='spacy',
            tokenizer_language='en_core_web_sm',
            lower=True,
            batch_first=True,
            include_lengths=True
        )
        answers = legacy.data.Field(sequential=False)

        # make splits for data
        try:
            train, dev, test = legacy.datasets.SNLI.splits(
                text_field=inputs, label_field=answers
            )
        except OSError:
            from io import BytesIO

            # filebytes = BytesIO(get_zip_data())
            # myzipfile = zipfile.ZipFile(filebytes)
            # for name in myzipfile.namelist():
            #     [ ... ]
            with zipfile.ZipFile(".data/snli/snli_1.0.zip", "r") as zip_ref:
                members2extract = ['snli_1.0/',  'snli_1.0/snli_1.0_dev.jsonl', 'snli_1.0/snli_1.0_dev.txt', 'snli_1.0/snli_1.0_test.jsonl', 'snli_1.0/snli_1.0_test.txt', 'snli_1.0/snli_1.0_train.jsonl', 'snli_1.0/snli_1.0_train.txt']
                zip_ref.extractall(DATA_PATH, members2extract)
            train, dev, test = legacy.datasets.SNLI.splits(
                text_field=inputs, label_field=answers
            )

        inputs.build_vocab(train, min_freq=1, vectors="glove.6B.300d")
        answers.build_vocab(train)
        save_cache(list(train), list(dev), list(test), inputs, answers)
    return (train, dev, test), inputs, answers



def glov_dict():
    glov_dir = wget.download("http://nlp.stanford.edu/data/{}".format('glove.6B.zip'))
    zip = zipfile.ZipFile(glov_dir)
    zip.extractall(path=".")

    glov = {}
    with open("./data/glove.6B.300d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            glov[word] = torch.from_numpy(vector)
            glov[word] = glov[word].type(torch.float)
    return glov


def get_glove_vector(sentences, F2I):
    index_sent = 0
    sentences_ret = []
    for sent in sentences:
        sentence = []
        index_word = 0
        for word in sent:
            if word in F2I:
                sentence.append(F2I[word].reshape(1, -1))
            else:
                sentence.append(F2I[UNIQUE].reshape(1, -1))
            index_word += 1
        sentence = torch.cat(sentence)
        sentence = sentence.reshape(1, sentence.shape[0], sentence.shape[1])
        sentences_ret.append(sentence)
        index_sent += 1
    return torch.cat(sentences_ret)

import time
start = time.time()
load_snli()
end = time.time()
print(end - start)