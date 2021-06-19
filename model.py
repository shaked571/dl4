from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class BiLSTM(nn.Module):
    def __init__(self, pre_trained_emb, hidden_dim: int, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=pre_trained_emb, freeze=True
        )

        self.embed_dim = self.embedding.embedding_dim #TODO verify
        self.dropout_val = dropout

        self.blstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            bidirectional=True)
        # self.linear = nn.Linear(2*hidden_dim, self.vocab.num_of_labels)

    def forward(self, x, x_lens):
        embeds = self.embedding(x)
        x_packed = pack_padded_sequence(embeds, x_lens, batch_first=True, enforce_sorted=False)
        out, (last_hidden_state, c_n) = self.blstm(x_packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        # out = self.linear(out)
        # out = out.flatten(0, 1)
        return out


class InnerAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super(InnerAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.w_y = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.w_h = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        x_mean = torch.mean(x, dim=1)


class Siamese(nn.Module):
    def __init__(self, pre_trained_emb, hidden_dim: int, dropout=0.2):
        super(Siamese, self).__init__()
        self.bilstm = BiLSTM(pre_trained_emb, hidden_dim, dropout)
        self.inner_attention = InnerAttention(hidden_dim=hidden_dim)

    def forward(self, prem, prem_lens, hyp, hyp_lens):

        prem_vec = self.bilstm(prem, prem_lens)
        hyp_vec = self.bilstm(hyp, hyp_lens)

        prem_atten = self.inner_attention(prem_vec)
        hyp_atten = self.inner_attention(hyp_vec)
