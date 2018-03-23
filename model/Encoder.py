import torch
import torch.nn as nn
from torch.autograd import Variable
import logging

from utils import *


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bidir = True
        self.gru = nn.GRU(embed_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.bidir, )

        self.console_logger = logging.getLogger()

    def forward(self, source, src_length=None, hidden=None):
        """
        source: B x T
        """
        batch_size = source.size(0)
        src_embed = self.embedding(source)

        self.console_logger.debug("encoder src_embed:  %1.3f", torch.sum(src_embed.data))

        if hidden is None:
            h_size = (self.num_layers * 2, batch_size, self.hidden_dim)
            enc_h_0 = Variable(src_embed.data.new(*h_size).normal_(), requires_grad=False)

        if src_length is not None:
            src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, src_length, batch_first=True)

        enc_h, enc_h_t = self.gru(src_embed, enc_h_0)

        # enc_h, _ = nn.utils.rnn.pad_packed_sequence(enc_h, batch_first=True)

        try:
            last_state_index = 2 if self.bidir else 1
        except AttributeError:
            last_state_index = 1

        last_layer_state = enc_h_t.transpose(0, 1)[:, -last_state_index:, :]

        self.console_logger.debug("encoder source:  %1.3f", torch.sum(source.data))
        self.console_logger.debug("encoder last_layer_state:  %1.3f", torch.sum(last_layer_state.data))

        return enc_h, last_layer_state
