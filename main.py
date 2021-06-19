import os
from typing import List, Optional
import torch.nn as nn
# from vocab import Vocab, SeqVocab
import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from data_loader import load_snli
from model import Siamese
ORIGINAL = 'o'
WITH_XAVIER = 'x'
def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=-100)

    return xx_pad, yy_pad, x_lens, y_lens

def model_xavier():
    pass

class Trainer:
    def __init__(self, how2run=ORIGINAL, n_ep=1,lr=0.001, steps_to_eval=500):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train,self.dev, self.test, self.inputs, self.answers = load_snli()
        if how2run == ORIGINAL:
            self.model = Siamese()
            self.lr = lr
            self.optimizer = optim.RMSprop(self.model.parameters(), selr=self.lr)

        elif how2run == WITH_XAVIER:
            self.model = nn.LSTM()#TODO just for  example
            self.optimizer = optim.AdamW(self.model.parameters(),  lr=lr, weight_decay=1e-4)
        else:
            raise ValueError()
        self.dev_batch_size = 4048

        self.steps_to_eval = steps_to_eval
        self.n_epochs = n_ep
        self.loss_func = nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.model_args = {"lr": self.lr}
        output_path = self.suffix_run()
        if not os.path.isdir(output_path):
            os.mkdir('outputs')
        self.saved_model_path = os.path.join('outputs',f"{output_path}.bin")
        self.writer = SummaryWriter(log_dir=f"tensor_board/{output_path}")
        self.best_model = None
        self.best_score = 0

if __name__ == '__main__':
    trainer = Trainer()
    print(trainer.data)