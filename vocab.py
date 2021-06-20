import abc
from collections import Counter


class Vocab:
    UNKNOWN_TOKEN = "UUUNKKK"
    PAD_DUMMY = "PAD_DUMMY"
    PAD_IDX = 0

    def __init__(self, inputs, taret):
        self.tokens = inputs.vocab

        self.labels = self.get_unique()
        self.tokens.insert(self.PAD_IDX, self.PAD_DUMMY)
        self.vocab_size = len(self.tokens)
        self.num_of_labels = len(self.labels)
        self.i2token = {i: w for i, w in enumerate(self.tokens)}
        self.token2i = {w: i for i, w in self.i2token.items()}
        self.i2label = {i: l for i, l in enumerate(self.labels)}
        self.label2i = {l: i for i, l in self.i2label.items()}
        self.vectors = [] #TODO

    def get_word_index(self, word):
        if word in self.token2i:
            return self.token2i[word]
        return self.token2i[self.UNKNOWN_TOKEN]

    @abc.abstractmethod
    def get_unique(self):
        pass


