import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import Decoder
from model import Encoder
from model import *


class Seq2Seq(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer, embed_dim, hidden_dim, max_len, trg_soi):
        super(Seq2Seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword

        self.encoder = Encoder(src_nword, embed_dim, hidden_dim)
        self.decoder = Decoder(trg_nword, embed_dim, hidden_dim)

    def forward(self, source, src_length=None, target=None, trg_length=None):
        batch_size = source.size(0)

        enc_h, enc_h_t = self.encoder(source, src_length)
        dec_h, dec_h_t = self.decoder(target, trg_length)

        bi_enc_h_t = torch.sum(enc_h_t, dim=1)
        bi_dec_h_t = torch.sum(dec_h_t, dim=1)

        bi_enc_h_t = bi_enc_h_t.div(bi_enc_h_t.norm(p=2, dim=1, keepdim=True).expand_as(bi_enc_h_t))
        bi_dec_h_t = bi_dec_h_t.div(bi_dec_h_t.norm(p=2, dim=1, keepdim=True).expand_as(bi_dec_h_t))

        loss = torch.mm(bi_enc_h_t, bi_dec_h_t.transpose(0, 1))
        for x in range(0, loss.size()[0]):
            loss[x, x] = - 10 * loss[x, x]
        loss = torch.sum(torch.sigmoid(loss))

        return enc_h_t, enc_h_t, dec_h_t, loss
