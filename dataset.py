from torch.utils.data import Dataset
import torch


class SNLIDataSet(Dataset):
    def __init__(self, data,vocab ):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def get_tensor(self, sent):
        res = []
        for w in sent:
            if w in self.input_info.vocab.stoi:
                res.append(self.vocab.stoi[w])
            else:
                res.append(self.vocab.stoi[self.input_info.unk_token])
        return torch.tensor(res).to(torch.int64)

    def __getitem__(self, index):
        premise = self.data[index].premise
        hyp = self.data[index].hypothesis
        label = self.data[index].label
        hyp_tensor = self.get_tensor(hyp)
        premise_tensor = self.get_tensor(premise)
        label_tensor = torch.tensor([self.vocab.stoi[label]]).to(torch.int64)

        return hyp_tensor, premise_tensor, label_tensor
