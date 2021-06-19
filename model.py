from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab: Vocab, dropout=0.2, sent_len=128):
        super(BiLSTM, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.vocab_size = self.vocab.vocab_size
        self.sent_len = sent_len
        self.embed_dim = embedding_dim
        self.dropout_val = dropout
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0) # TODO: don't update the weights

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
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab: Vocab):
        super(InnerAttention, self).__init__()
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab.vocab_size
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.w_y = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.w_h = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        x_mean = torch.mean(x, dim=1)


class Siamese(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab: Vocab, dropout=0.2, sent_len=128):
        super(Siamese, self).__init__()
        self.bilstm = BiLSTM(embedding_dim, hidden_dim, vocab, dropout, sent_len)
        self.inner_attention = InnerAttention(embedding_dim=embedding_dim,
                                              hidden_dim=hidden_dim,
                                              vocab=vocab)

    def forward(self, prem, prem_lens, hyp, hyp_lens):
        prem_vec = self.bilstm(prem, prem_lens)
        hyp_vec = self.bilstm(hyp, hyp_lens)

        prem_atten = self.inner_attention(prem_vec)
        hyp_atten = self.inner_attention(hyp_vec)
