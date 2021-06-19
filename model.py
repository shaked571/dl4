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

        self.embed_dim = self.embedding.embedding_dim
        self.dropout_val = dropout
        self.dropout = nn.Dropout(dropout)

        self.blstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            bidirectional=True)

    def forward(self, x):
        output = self.embedding(x)
        output = self.dropout(output)
        output = self.relu(output) #TODO need?? read paper
        output = output.transpose(0, 1)  # make it (seq_len, batch_size, features)
        output, (hidden, cell) = self.lstm(output.unsqueeze(0))
        return output, (hidden, cell)


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

    def forward(self, prem, hyp):

        prem_vec, (hidden, cell) = self.bilstm(prem)
        hyp_vec, (hidden, cell) = self.bilstm(hyp)

        prem_atten = self.inner_attention(prem_vec)
        hyp_atten = self.inner_attention(hyp_vec)
