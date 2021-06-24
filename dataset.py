from torch.utils.data import Dataset
import torch


class SNLIDataSet(Dataset):
    def __init__(self, data, input_info, labels_info):
        self.data = data
        self.input_info = input_info
        self.labels_info = labels_info

    def __len__(self):
        return len(self.data)

    def get_tensor(self, sent):
        res = []
        for w in sent:
            if w in self.input_info.vocab.stoi:
                res.append(self.input_info.vocab.stoi[w])
            else:
                res.append(self.input_info.vocab.stoi[self.input_info.unk_token])
        return torch.tensor(res).to(torch.int64)

    def remove_intersection(self, hyp, prem):
        new_hyp = []
        for word in hyp:
            if word in prem:
                prem.remove(word)
            else:
                new_hyp.append(word)
        return new_hyp, prem

    def __getitem__(self, index):
        premise = self.data[index].premise
        hyp = self.data[index].hypothesis
        label = self.data[index].label
        hyp, premise = self.remove_intersection(hyp, premise)

        hyp_tensor = self.get_tensor(hyp)
        premise_tensor = self.get_tensor(premise)
        label_tensor = torch.tensor([self.labels_info.vocab.stoi[label]]).to(torch.int64)

        return hyp_tensor, premise_tensor, label_tensor
