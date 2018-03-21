import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from model import *

'''
https://github.com/allenai/allennlp
1. Encoder_base.py
- "this class provides functionality for sorting sequences by length" - gets directed to nn.util.py
2. PytorchSeq2VecWrapper.py
- get last state of the lstm
3. nn.util.py - sort_batch_by_length
'''


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bidir = True
        self.gru = nn.GRU(embed_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.bidir, )

    def forward(self, target, trg_length=None, hidden=None):
        batch_size = target.size(0)
        sorted_inputs, sorted_seq_len, restoration_indices, _ = sort_batch_by_length(target, Variable(torch.FloatTensor(trg_length)).cuda())

        src_embed = self.embedding(sorted_inputs)

        if hidden is None:
            h_size = (self.num_layers *2, batch_size, self.hidden_dim)
            enc_h_0 = Variable(src_embed.data.new(*h_size).zero_(), requires_grad=False)

        print("bug enter")
        seq_length = [int(x) for x in sorted_seq_len.data.tolist()]
        print("bug fixed")

        src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, seq_length, batch_first=True)
        enc_h, enc_h_t = self.gru(src_embed, enc_h_0)

        # enc_h, _ = nn.utils.rnn.pad_packed_sequence(enc_h, batch_first=True)

        unsorted_state = enc_h_t.transpose(0, 1).index_select(0, restoration_indices)

        try:
            last_state_index = 2 if self.bidir else 1
        except AttributeError:
            last_state_index = 1

        last_layer_state = unsorted_state[:, -last_state_index:, :]

        return enc_h, last_layer_state
