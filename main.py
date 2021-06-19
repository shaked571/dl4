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
import torchtext
import torchtext.legacy as legacy
class Trainer:
    def __init__(self, hidden_dim=100, dropout=0.2, sent_len=128, n_ep=1, lr=0.001, how2run=ORIGINAL,
                 steps_to_eval=500):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train, self.dev, self.test, self.inputs, self.answers = load_snli()
        self.train, self.dev, self.test = legacy.data.BucketIterator.splits((self.train, self.dev, self.test), batch_size=3, device=0)
        # self.train, self.dev, self.test = torchtext.data.datasets.SNLI.iter(batch_size=128)
        self.word2i = self.inputs.vocab.stoi
        self.embedding_vectors = self.inputs.vocab.vectors
        if how2run == ORIGINAL:
            self.model = Siamese(self.embedding_vectors, hidden_dim, dropout, sent_len)
            self.lr = lr
            self.optimizer = optim.RMSprop(self.model.parameters(), selr=self.lr)

        elif how2run == WITH_XAVIER:
            self.model = nn.LSTM() # TODO - just for example
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
        self.saved_model_path = os.path.join('outputs', f"{output_path}.bin")
        self.writer = SummaryWriter(log_dir=f"tensor_board/{output_path}")
        self.best_model = None
        self.best_score = 0

    def train(self):
        num_samples = 0
        for epoch in range(self.n_epochs):
            ###################
            # train the model #
            ###################
            print(f"start epoch: {epoch + 1}")
            train_loss = 0.0
            step_loss = 0
            self.model.train()  # prep model for training
            for step, example in tqdm(enumerate(self.train), total=len(self.train)):
                premise = example.premise
                hyp = example.hypothesis
                label = example.label
                num_samples += self.train.batch_size
                data = data.to(self.device)
                target = target.to(self.device)
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                self.model.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data, data_lens)  # Eemnded Data Tensor size (1,5)
                # calculate the loss
                loss = self.loss_func(output, target.view(-1))
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item() * data.size(0)
                step_loss += loss.item() * data.size(0)
                if num_samples >= self.steps_to_eval:
                    num_samples = 0
                    print(f"in step: {(step+1)*self.train.batch_size} train loss: {step_loss}")
                    self.writer.add_scalar('Loss/train_step', step_loss, step * (epoch + 1))
                    step_loss = 0.0
                    # print((step+1)*self.train_data.batch_size + epoch * len(self.train_data))
                    self.evaluate_model((step+1)*self.train.batch_size + epoch * len(self.train), "step", self.dev_data)
            print(f"in epoch: {epoch + 1} train loss: {train_loss}")
            self.writer.add_scalar('Loss/train', train_loss, epoch+1)
            print((epoch+1) * len(self.train) *self.train.batch_size)
            self.evaluate_model((epoch+1) * len(self.train)*self.train.batch_size, "epoch", self.dev)

    def evaluate_model(self, step, stage, data_set,write=True):
        with torch.no_grad():
            self.model.eval()
            loss = 0

            prediction = []
            all_target = []
            for eval_step, (data, target, data_lens, target_lens) in tqdm(enumerate(data_set), total=len(data_set),
                                                  desc=f"dev step {step} loop"):
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data, data_lens)  # Eemnded Data Tensor size (1,5)

                loss = self.loss_func(output.detach(), target.view(-1))
                loss += loss.item() * data.size(0)
                _, predicted = torch.max(output, 1)
                prediction += predicted.tolist()
                all_target += target.view(-1).tolist()
            accuracy = self.accuracy_token_tag(prediction, all_target)
            if write:
                print(f'Accuracy/dev_{stage}: {accuracy}')

                self.writer.add_scalar(f'Accuracy/dev_{stage}', accuracy, step)
                self.writer.add_scalar(f'Loss/dev_{stage}', loss, step)
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    torch.save(self.model.state_dict(), self.saved_model_path)

            else:
                print(f'Accuracy/train_{stage}: {accuracy}')

        self.model.train()

    def suffix_run(self):
        res = ""
        for k, v in self.model_args.items():
            res += f"{k}_{v}_"
        res = res.strip("_")
        return res


if __name__ == '__main__':
    trainer = Trainer()
    # print(trainer.data)
