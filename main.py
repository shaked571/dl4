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
from dataset import SNLIDataSet
from model import Siamese
ORIGINAL = 'o'
WITH_XAVIER = 'x'


def model_xavier():
    pass


class Trainer:
    def __init__(self, hidden_dim=100, dropout=0.2, n_ep=10, lr=0.001, how2run=ORIGINAL, steps_to_eval=50000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_raw, dev_raw, test_raw, self.inputs_info, self.labels_info = load_snli()
        self.batch_size = 128
        # self.word2i = self.inputs_info.vocab.stoi
        train_set = SNLIDataSet(train_raw,  self.inputs_info, self.labels_info)
        dev_set = SNLIDataSet(dev_raw,  self.inputs_info, self.labels_info)
        test_set = SNLIDataSet(test_raw,  self.inputs_info, self.labels_info)

        self.train_d = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.pad_collate)
        self.dev_d = DataLoader(dev_set, batch_size=self.batch_size, collate_fn=self.pad_collate)
        self.test_d = DataLoader(test_set, batch_size=self.batch_size, collate_fn=self.pad_collate)

        self.embedding_vectors = self.inputs_info.vocab.vectors
        if how2run == ORIGINAL:
            self.model: Siamese = Siamese(self.embedding_vectors, hidden_dim, dropout)
            self.lr = lr
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)

        elif how2run == WITH_XAVIER:
            self.model: Siamese = Siamese(self.embedding_vectors, hidden_dim, dropout) # TODO - just for example - need to inherit and runover
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
        if not os.path.isdir('outputs'):
            os.mkdir('outputs')
        self.saved_model_path = os.path.join('outputs', f"{output_path}.bin")
        self.writer = SummaryWriter(log_dir=f"tensor_board/{output_path}")
        self.best_model = None
        self.best_score = 0

    def pad_collate(self, batch):
        (s1, s2, l) = zip(*batch)
        sent1_lens = [len(sent) for sent in s1]
        sent2_lens = [len(sent) for sent in s2]

        s1_pad = pad_sequence(s1, batch_first=True, padding_value=1)
        s2_pad = pad_sequence(s2, batch_first=True, padding_value=1)

        return s1_pad.to(self.device), s2_pad.to(self.device), sent1_lens, sent2_lens, torch.Tensor(l).to(self.device).to(torch.int64)

    def train(self):
        num_samples = 0
        for epoch in range(self.n_epochs):
            print(f"start epoch: {epoch + 1}")
            train_loss = 0.0
            step_loss = 0
            self.model.train()  # prep model for training
            for step, (s1, s2, sent1_lens, sent2_lens, target) in tqdm(enumerate(self.train_d), total=len(self.train_d)):
                num_samples += self.batch_size
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                self.model.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(s1, s2, sent1_lens, sent2_lens)
                loss = self.loss_func(output, target.view(-1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * s1.size(0)
                step_loss += loss.item() * s1.size(0)
                if num_samples >= self.steps_to_eval:
                    num_samples = 0
                    print(f"in step: {(step+1)*self.batch_size} train loss: {step_loss}")
                    self.writer.add_scalar('Loss/train_step', step_loss, step * (epoch + 1))
                    step_loss = 0.0
                    # print((step+1)*self.train_data.batch_size + epoch * len(self.train_data))
                    self.evaluate_model((step+1)*self.batch_size + epoch * len(self.train_d), "step", self.dev_d)
            print(f"in epoch: {epoch + 1} train loss: {train_loss}")
            self.writer.add_scalar('Loss/train', train_loss, epoch+1)
            print((epoch+1) * len(self.train_d) * self.batch_size)
            self.evaluate_model((epoch+1) * len(self.train_d)*self.batch_size, "epoch", self.dev_d)

    def evaluate_model(self, step, stage, data_set,save_model=True):
        with torch.no_grad():
            self.model.eval()
            loss = 0

            prediction = []
            all_target = []
            total = 0
            correct = 0
            for s1, s2, sent1_lens, sent2_lens, target in tqdm(data_set, total=len(data_set),
                                                               desc=f"dev step {step} loop"):
                output = self.model(s1, s2, sent1_lens, sent2_lens)
                loss = self.loss_func(output.detach(), target.view(-1))
                loss += loss.item() * s1.size(0)
                _, predicted = torch.max(output, 1)
                prediction += predicted.tolist()
                all_target += target.view(-1).tolist()
                total += target.size(0)
                correct += (predicted == target).sum().item()

            accuracy = 100 * correct / total
            print(f'Accuracy/dev_{stage}: {accuracy}')

            self.writer.add_scalar(f'Accuracy/dev_{stage}', accuracy, step)
            self.writer.add_scalar(f'Loss/dev_{stage}', loss, step)
            if accuracy > self.best_score and save_model:
                self.best_score = accuracy
                torch.save(self.model.state_dict(), self.saved_model_path)
        self.model.train()

    def suffix_run(self):
        res = ""
        for k, v in self.model_args.items():
            res += f"{k}_{v}_"
        res = res.strip("_")
        return res

    def test(self):
        self.model.load_model(self.saved_model_path)
        self.evaluate_model(1, 'test', trainer.test_d, save_model=False)



if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    trainer.test()
