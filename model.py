from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class BiLSTM(nn.Module):
    def __init__(self, pre_trained_emb, hidden_dim: int, dropout, drop_lstm, drop_embedding):
        super(BiLSTM, self).__init__()
        self.drop_lstm = drop_lstm
        self.drop_embedding = drop_embedding

        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=pre_trained_emb, freeze=True
        )

        self.embed_dim = self.embedding.embedding_dim
        self.dropout_val = dropout
        self.dropout = nn.Dropout(dropout)
        self.num_layers = 1
        self.bilstm = nn.LSTM(input_size=self.embed_dim,
                              hidden_size=hidden_dim,
                              num_layers= self.num_layers,
                              batch_first=True,
                              bidirectional=True,
                              dropout=self.dropout_val if self.drop_lstm and self.num_layers > 1 else 0
                              )

    def forward(self, x, x_lens):
        output = self.embedding(x)
        if self.drop_embedding:
            output = self.dropout(output)

        x_packed = pack_padded_sequence(output, x_lens, batch_first=True, enforce_sorted=False)
        output, _ = self.bilstm(x_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output


class InnerAttention(nn.Module):
    def __init__(self, pre_trained_emb, hidden_dim: int, dropout: float, drop_lstm: bool, drop_embedding: bool, xavier:bool, device):
        super(InnerAttention, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.tanh = nn.Tanh()
        self.bilstm = BiLSTM(pre_trained_emb, hidden_dim, dropout, drop_lstm, drop_embedding)
        self.softmax = nn.Softmax(dim=1)
        self.lstm_out_dim = 2 * self.hidden_dim
        self.attention_dim = int(self.hidden_dim * 1.7)
        self.w_y = nn.Linear(self.lstm_out_dim, self.attention_dim)
        self.w_h = nn.Linear(self.lstm_out_dim, self.attention_dim)
        self.w = nn.Linear(self.attention_dim, 1)
        if xavier:
            nn.init.xavier_uniform_(self.w_y.weight)
            nn.init.xavier_uniform_(self.w_h.weight)
            nn.init.xavier_uniform_(self.w.weight)

    def forward(self, x, x_lens):
        y = self.bilstm(x, x_lens)
        r_avg = torch.mean(y, dim=1).unsqueeze(1)
        r_avg = r_avg.permute(0, 2, 1)
        r_avg_e_l = torch.matmul(r_avg, torch.ones([1, y.shape[1]]).to(self.device)).permute(0, 2, 1)
        m = self.tanh(self.w_y(y) + self.w_h(r_avg_e_l))
        alpha = self.softmax(self.w(m))
        r_att = torch.bmm(y.permute(0, 2, 1), alpha).permute(0, 2, 1)
        return r_att


class Siamese(nn.Module):
    def __init__(self, pre_trained_emb, hidden_dim: int, dropout, drop_lstm: bool, drop_embedding: bool, xavier: bool, device):
        super(Siamese, self).__init__()
        self.device = device
        self.inner_attention = InnerAttention(pre_trained_emb=pre_trained_emb,
                                              hidden_dim=hidden_dim,
                                              dropout=dropout,
                                              drop_lstm=drop_lstm,
                                              drop_embedding=drop_embedding,
                                              xavier=xavier,
                                              device=device)
        self.linear1 = nn.Linear(8 * hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.linear_predictor = nn.Linear(int(hidden_dim/2), 3)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, prem, hyp, prem_lens, hyp_lens):
        prem_atten = self.inner_attention(prem, prem_lens)
        hyp_atten = self.inner_attention(hyp, hyp_lens)
        mult_vec = prem_atten * hyp_atten
        diff_vec = prem_atten - hyp_atten
        concat_vec = torch.cat([prem_atten, mult_vec, diff_vec, hyp_atten], dim=2)
        concat_vec = self.dropout(concat_vec)
        y = self.linear1(concat_vec)
        y = self.tanh(y)
        y = self.linear2(y)
        y = self.tanh(y)
        y = self.linear_predictor(y).squeeze(1)
        y = self.tanh(y)
        return y

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint)

