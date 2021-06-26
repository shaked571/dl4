import argparse
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from data_loader import load_snli
from dataset import SNLIDataSet
from model import Siamese
import random
import numpy as np
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def set_seed(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() == 'cuda':
        torch.cuda.manual_seed_all(seed)


def model_xavier():
    pass


class Trainer:
    def __init__(self, drop_lstm: bool, drop_embedding: bool, xavier: bool, adamw:bool, hidden_dim=235, dropout=0.25,
                 n_ep=6, diff=False, relu=False, lr=0.001, steps_to_eval=50000, batch=128, gpu=0, seed=1):
        self.device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else "cpu"
        print(self.device)
        train_raw, dev_raw, test_raw, self.inputs_info, self.labels_info = load_snli()
        self.train_batch_size = batch
        self.dev_batch_size = 1000
        train_set = SNLIDataSet(train_raw,  self.inputs_info, self.labels_info, diff)
        dev_set = SNLIDataSet(dev_raw,  self.inputs_info, self.labels_info, diff)
        test_set = SNLIDataSet(test_raw,  self.inputs_info, self.labels_info, diff)
        self.train_d = DataLoader(train_set, batch_size=self.train_batch_size, collate_fn=self.pad_collate)
        self.dev_d = DataLoader(dev_set, batch_size=self.dev_batch_size, collate_fn=self.pad_collate)
        self.test_d = DataLoader(test_set, batch_size=self.dev_batch_size, collate_fn=self.pad_collate)
        self.inputs_info = self.update_unk_vec(self.inputs_info)
        self.embedding_vectors = self.inputs_info.vocab.vectors
        self.lr = lr
        self.model: Siamese = Siamese(self.embedding_vectors, hidden_dim, dropout, drop_lstm, drop_embedding, xavier, device=self.device, use_relu=relu)
        if adamw:
            self.optimizer = optim.AdamW(self.model.parameters(),  lr=self.lr, weight_decay=1e-4)
        else:
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)

        self.dev_batch_size = 4048

        self.steps_to_eval = steps_to_eval
        self.n_epochs = n_ep
        self.loss_func = nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.model_args = {"drop_lstm": drop_lstm, "drop_embedding": drop_embedding, "xav": xavier, "relu": relu,
                           "diff": diff, "adamw": adamw, "epoch": n_ep, "seed": seed, "lr": lr, "batch": batch,
                           'dropout': dropout}
        output_path = self.suffix_run()
        if not os.path.isdir('outputs'):
            os.mkdir('outputs')
        self.saved_model_path = os.path.join('outputs', f"{output_path}.bin")
        self.writer = SummaryWriter(log_dir=f"tensor_board/{output_path}")
        self.best_model = None
        self.best_score = 0

    def update_unk_vec(self, inputs_info):
        unk_vec = torch.Tensor(np.random.uniform(-0.05, 0.05, 300)).to(torch.float32)
        inputs_info.vocab.vectors[self.inputs_info.vocab.unk_index] = unk_vec
        return  inputs_info

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
                num_samples += self.train_batch_size
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
                    print(f"in step: {(step+1)*self.train_batch_size} train loss: {step_loss}")
                    self.writer.add_scalar('Loss/train_step', step_loss, step * (epoch + 1))
                    step_loss = 0.0
                    # print((step+1)*self.train_data.batch_size + epoch * len(self.train_data))
                    self.evaluate_model((step+1) * self.train_batch_size + epoch * len(self.train_d.dataset), "epoch", self.dev_d)
            print(f"in epoch: {epoch + 1} train loss: {train_loss}")
            self.writer.add_scalar('Loss/train', train_loss, epoch+1)
            print((epoch+1) * len(self.train_d) * self.train_batch_size)
            self.evaluate_model((epoch+1) * len(self.train_d.dataset), "epoch", self.dev_d)
            self.evaluate_model((epoch+1) * len(self.train_d.dataset), "train_epoch", self.train_d)
        self.writer.flush()

    def evaluate_model(self, step, stage, data_set, save_model=True):
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
            self.writer.add_scalar(f'Accuracy/{stage}', accuracy, step)
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

    def test(self, path=None):
        if path is None:
            path = self.saved_model_path
        self.model.load_model(path)
        self.evaluate_model(1, 'test', self.test_d, save_model=False)
        self.writer.flush()
        self.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aligner model')
    parser.add_argument('-ld', '--lstm_drop', help='lstm dropout', action='store_true')
    parser.add_argument('-le', '--lstm_embedding', help='embedding dropout', action='store_true')
    parser.add_argument('-r', '--relu', help='to use relu', action='store_true')
    parser.add_argument('-d', '--difference', help='to use difference words between hyp anf prem', action='store_true')
    parser.add_argument('-x', '--xavier', help='to use xavier init', action='store_true')
    parser.add_argument('-a', '--adamw', help='to use adamW instead of RMSprop ', action='store_true')
    parser.add_argument('-e', '--epoch', help='embedding dropout', type=int, default=10, required=False)
    parser.add_argument('-s', '--seed', help='seed', type=int, default=1, required=False)
    parser.add_argument('-l', '--lr', help='learning rate', type=float, default=0.001, required=False)
    parser.add_argument('-b', '--batch', help='train batch size', type=int, default=128, required=False)
    parser.add_argument('-do', '--drop', help='train batch size', type=float, default=0.25, required=False)
    parser.add_argument('--gpu', type=int, default=0, required=False)
    args = parser.parse_args()
    print(args.lr)
    set_seed(args.seed)
    trainer = Trainer(args.lstm_drop, args.lstm_embedding, args.xavier, args.adamw, n_ep=args.epoch,
                      diff=args.difference, relu=args.relu, gpu=args.gpu, batch=args.batch, lr=args.lr, seed=args.seed,
                      dropout=args.drop)

    model_parameters = filter(lambda p: p.requires_grad, trainer.model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    count_parameters(trainer.model)

    trainer.train()
    trainer.test()
