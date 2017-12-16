import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import LSTM, Linear, Module, Embedding, Parameter

from net.util import PADDING_INDEX, EOS_INDEX, SOS_INDEX


def len_unpadded(x):
    return next((i for i, j in enumerate(x) if scalar(j) == 0), len(x))


def scalar(x):
    return x.view(-1).data.tolist()[0]


def argmax(x):
    return scalar(torch.max(x, 0)[1])


def log_sum_exp(x):
    max_score = x[argmax(x)]
    max_score_broadcast = max_score.expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast)))


def log_sum_exp_batch1(x):
    max_score = torch.cat([i[argmax(i)] for i in x])
    max_score_broadcast = max_score.view(-1, 1).expand_as(x)
    z = max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), 1))
    return z


def log_sum_exp_batch2(x, y):
    z = x[:len(y)] + y
    max_score = torch.cat([i[argmax(i)] for i in z])
    max_score_broadcast = max_score.view(-1, 1).expand_as(z)
    z = max_score + torch.log(torch.sum(torch.exp(z - max_score_broadcast), 1))
    if len(x) > len(z):
        z = torch.cat((z, torch.cat([i[argmax(i)] for i in x[len(y):]])))
    return z.view(len(x), 1)


class BiLSTMWithCRF(Module):
    def __init__(self, vocabs_size: int, tags_size: int, batch_size: int, preinit_word_embedding=None,
                 hidden_dim: int = 512):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.tags_size = tags_size
        self.embed = Embedding(vocabs_size, 300, padding_idx=PADDING_INDEX)
        if preinit_word_embedding is not None:
            self.embed.weight.data.copy_(preinit_word_embedding)
        self.lstm = LSTM(input_size=300,
                         hidden_size=hidden_dim,
                         num_layers=2,
                         dropout=0.5,
                         bias=True,
                         batch_first=True,
                         bidirectional=True)
        self.out = Linear(2 * hidden_dim, tags_size)
        self.trans = Parameter(torch.randn(self.tags_size, self.tags_size))
        self.trans.data[SOS_INDEX, :] = -10000.
        self.trans.data[:, EOS_INDEX] = -10000.
        self.trans.data[:, PADDING_INDEX] = -10000.
        self.trans.data[PADDING_INDEX, :] = -10000.
        self.trans.data[PADDING_INDEX, EOS_INDEX] = 0.
        self.trans.data[PADDING_INDEX, PADDING_INDEX] = 0.
        self.hidden = self.init_hidden()

    def lstm_forward(self, x):
        self.hidden = self.init_hidden()
        self.lengths = sorted([len_unpadded(seq) for seq in x], reverse=True)
        embed = self.embed(x)
        embed = nn.utils.rnn.pack_padded_sequence(embed, self.lengths, batch_first=True)
        y, _ = self.lstm(embed, self.hidden)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True)
        y = self.out(y)
        return y

    def init_hidden(self):  # initialize hidden states
        h = Variable(torch.zeros(2, self.batch_size, self.hidden_dim))  # hidden states
        c = Variable(torch.zeros(2, self.batch_size, self.hidden_dim))  # cell states
        return h, c

    def crf_score(self, y, y0):
        score = Variable(torch.FloatTensor(self.batch_size).cuda().fill_(0.))
        y0 = torch.cat([torch.LongTensor(self.batch_size, 1).cuda().fill_(SOS_INDEX), y0], 1)
        for b in range(len(self.lengths)):
            for t in range(self.lengths[b]):  # iterate through the sequence
                emit_score = y[b, t, y0[b, t + 1]]
                trans_score = self.trans[y0[b, t + 1], y0[b, t]]
                score[b] = score[b] + emit_score + trans_score
        return score

    def crf_score_batch(self, y, y0, mask):
        score = Variable(torch.FloatTensor(self.batch_size).fill_(0.).cuda())
        y0 = torch.cat([torch.LongTensor(self.batch_size, 1).cuda().fill_(SOS_INDEX), y0], 1)
        for t in range(y.size(1)):
            mask_t = Variable(mask[:, t])
            emit_score = torch.cat([y[b, t, y0[b, t + 1]] for b in range(self.batch_size)])
            trans_score = torch.cat([self.trans[seq[t + 1], seq[t]] for seq in y0]) * mask_t
            score = score + emit_score + trans_score
        return score

    def crf_forward(self, y):
        score = torch.FloatTensor(self.batch_size, self.tags_size).fill_(-10000.).cuda()
        score[:, SOS_INDEX] = 0.
        score = Variable(score)
        for b in range(len(self.lengths)):
            for t in range(self.lengths[b]):
                score_t = []
                for f in range(self.tags_size):
                    emit_score = y[b, t, f].expand(self.tags_size)
                    trans_score = self.trans[f].expand(self.tags_size)
                    z = log_sum_exp(score[b] + emit_score + trans_score)
                    score_t.append(z)
                score[b] = torch.cat(score_t)
        score = torch.cat([log_sum_exp(i) for i in score])
        return score

    def crf_forward_batch(self, y, mask):
        score = torch.FloatTensor(self.batch_size, self.tags_size).fill_(-10000.).cuda()
        score[:, SOS_INDEX] = 0.
        score = Variable(score)
        for t in range(y.size(1)):
            score_t = []
            len_t = int(torch.sum(mask[:, t]))
            for f in range(self.tags_size):
                emit_score = torch.cat([y[b, t, f].expand(1, self.tags_size) for b in range(len_t)])
                trans_score = self.trans[f].expand(len_t, self.tags_size)
                z = log_sum_exp_batch2(score, emit_score + trans_score)
                score_t.append(z)
            score = torch.cat(score_t, 1)
        score = log_sum_exp_batch1(score).view(self.batch_size)
        return score

    def viterbi(self, y):
        bptr = []
        score = torch.FloatTensor(self.tags_size).fill_(-10000.).cuda()
        score[SOS_INDEX] = 0.
        score = Variable(score)
        for t in range(len(y)):
            bptr_t = []
            score_t = []
            for i in range(self.tags_size):  # for each next tag
                z = score + self.trans[i]
                best_tag = argmax(z)  # find the best previous tag
                bptr_t.append(best_tag)
                score_t.append(z[best_tag])
            bptr.append(bptr_t)
            score = torch.cat(score_t) + y[t]
        best_tag = argmax(score)
        best_score = score[best_tag]
        best_path = [best_tag]
        for bptr_t in reversed(bptr):
            best_path.append(bptr_t[best_tag])
        best_path = reversed(best_path[:-1])

        return best_path

    def loss(self, x, y0):
        y = self.lstm_forward(x)
        mask = x.data.gt(0).float()
        y = y * Variable(mask.unsqueeze(-1).expand_as(y))
        score = self.crf_score_batch(y, y0, mask)
        Z = self.crf_forward_batch(y, mask)
        L = torch.mean(Z - score)
        return L

    def forward(self, x):
        result = []
        y = self.lstm_forward(x)
        for i in range(len(self.lengths)):
            if self.lengths[i] > 1:
                best_path = self.viterbi(y[i][:self.lengths[i]])
            else:
                best_path = []
            result.append(best_path)
        return result
