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

        self.bilstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, x, x_lens):
        output = self.embedding(x)
        output = self.dropout(output)
        x_packed = pack_padded_sequence(output, x_lens, batch_first=True, enforce_sorted=False)
        # output = self.relu(output) #TODO need?? read paper
        # output = output.transpose(0, 1)  # make it (seq_len, batch_size, features)
        # output, (hidden, cell) = self.bilstm(output.unsqueeze(0))
        output, _ = self.bilstm(x_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output


class InnerAttention(nn.Module):
    def __init__(self, pre_trained_emb, hidden_dim: int, dropout=0.2):
        super(InnerAttention, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_dim = hidden_dim
        self.tanh = nn.Tanh()
        self.bilstm = BiLSTM(pre_trained_emb, hidden_dim, dropout)
        self.softmax = nn.Softmax(dim=1)
        self.w_y = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.w_h = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
        self.w = nn.Linear(2 * self.hidden_dim, 1)

    def forward(self, x, x_lens):
        y = self.bilstm(x, x_lens)
        r_avg = torch.mean(y, dim=1, keepdim=True)
        r_avg = r_avg.permute(0, 2, 1)
        r_avg_e_l = torch.matmul(r_avg, torch.ones([1, y.shape[1]])).permute(0, 2, 1)
        m = self.tanh(self.w_y(y) + self.w_h(r_avg_e_l))
        alpha = self.softmax(self.w(m))
        r_att = torch.bmm(y.permute(0, 2, 1), alpha).permute(0, 2, 1)
        # r_att = y*alpha
        return r_att


class Siamese(nn.Module):
    def __init__(self, pre_trained_emb, hidden_dim: int, dropout=0.2):
        super(Siamese, self).__init__()
        self.inner_attention = InnerAttention(pre_trained_emb=pre_trained_emb,
                                              hidden_dim=hidden_dim,
                                              dropout=dropout)
        self.softmax = nn.Softmax(dim=1)
        self.linear_predictor = nn.Linear(8 * hidden_dim, 3)

    def forward(self, prem, hyp, prem_lens, hyp_lens):
        prem_atten = self.inner_attention(prem, prem_lens)
        hyp_atten = self.inner_attention(hyp, hyp_lens)

        mult_vec = prem_atten * hyp_atten
        diff_vec = prem_atten - hyp_atten
        concat_vec = torch.cat([prem_atten, mult_vec, diff_vec, hyp_atten], dim=2)
        y = self.linear_predictor(concat_vec).squeeze(1)
        return y
