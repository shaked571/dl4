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
        self.embedding = self.get_embedding_layer()

        self.blstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            bidirectional=True)
        self.linear = nn.Linear(2*hidden_dim, self.vocab.num_of_labels)

    def forward(self, x, x_lens):
        embeds = self.get_embedx_vectors(x, x_lens)
        x_packed = pack_padded_sequence(embeds, x_lens, batch_first=True, enforce_sorted=False)
        out, (last_hidden_state, c_n) = self.blstm(x_packed)
        out2, _ = pad_packed_sequence(out, batch_first=True)
        out3 = self.linear(out2)
        out4 = out3.flatten(0, 1)
        return out4


class InnerAttention(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab: Vocab, dropout=0.2, sent_len=128):
        super(InnerAttention, self).__init__()
        self.embed_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(input_size=self.embed_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=2,
                              bidirectional=True)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.w_y = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.w_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        # pool of non-square window
        self.pooling = nn.AvgPool2d((3, 2), stride=(2, 1))

    def get_bilstm_sent(self, sent, sent_lens):
        embeds = self.embedding(sent)
        x_packed = pack_padded_sequence(embeds, sent_lens, batch_first=True, enforce_sorted=False)
        out, (last_hidden_state, c_n) = self.bilstm(x_packed)
        bilstm_vec, _ = pad_packed_sequence(out, batch_first=True)
        return bilstm_vec

    def forward(self, prem, prem_lens, hyp, hyp_lens):
        prem_vec = self.get_bilstm_sent(prem, prem_lens)
        hyp_vec = self.get_bilstm_sent(hyp, hyp_lens)

        prem_mean = torch.mean(prem_vec, dim=1)
        hyp_mean = torch.mean(hyp_vec, dim=1)


