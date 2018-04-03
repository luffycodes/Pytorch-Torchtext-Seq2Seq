import torch
import torch.nn as nn
from torch.autograd import Variable
import logging

from utils import *


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, bi_dir):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, max_norm=1)
        self.bidir = bi_dir
        self.gru = nn.GRU(embed_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.bidir,
                          bias=True)

        self.console_logger = logging.getLogger()

    def forward(self, source, src_length=None, hidden=None, sts=False, sort=False):
        """
        source: B x T
        """
        if not sts:
            batch_size = source.size(0)
            src_embed = self.embedding(source)

            if hidden is None:
                h_size = (self.num_layers, batch_size, self.hidden_dim)
                enc_h_0 = Variable(src_embed.data.new(*h_size).zero_(), requires_grad=False)

            if src_length is not None:
                src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, src_length, batch_first=True)

            self.console_logger.debug("encoder src_embed:  %1.3f", torch.sum(src_embed.data))

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
        else:
            enc_h, last_layer_state = self.stsForward(source, src_length, hidden, sort)
            return enc_h, last_layer_state

    def stsForward(self, source, src_length=None, hidden=None, sort=False):
        if not sort:
            batch_size = source.size(0)
            src_embed = self.embedding(source)

            if hidden is None:
                h_size = (self.num_layers, batch_size, self.hidden_dim)
                enc_h_0 = Variable(src_embed.data.new(*h_size).zero_(), requires_grad=False)

            if src_length is not None:
                src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, src_length, batch_first=True)

            self.console_logger.debug("stsForward encoder src_embed:  %1.3f", torch.sum(src_embed.data))

            enc_h, enc_h_t = self.gru(src_embed, enc_h_0)

            # enc_h, _ = nn.utils.rnn.pad_packed_sequence(enc_h, batch_first=True)

            try:
                last_state_index = 2 if self.bidir else 1
            except AttributeError:
                last_state_index = 1

            last_layer_state = enc_h_t.transpose(0, 1)[:, -last_state_index:, :]

            self.console_logger.debug("stsForward encoder source:  %1.3f", torch.sum(source.data))
            self.console_logger.debug("stsForward encoder last_layer_state:  %1.3f", torch.sum(last_layer_state.data))
            return enc_h, last_layer_state
        else:
            batch_size = source.size(0)

            trg_length_cuda = Variable(torch.FloatTensor(src_length))
            if torch.cuda.is_available():
                trg_length_cuda = trg_length_cuda.cuda()

            sorted_inputs, sorted_seq_len, restoration_indices, _ = sort_batch_by_length(source, trg_length_cuda)

            src_embed = self.embedding(sorted_inputs)

            if hidden is None:
                h_size = (self.num_layers, batch_size, self.hidden_dim)
                enc_h_0 = Variable(src_embed.data.new(*h_size).zero_(), requires_grad=False)

            seq_length = [int(x) for x in sorted_seq_len.data.tolist()]

            src_embed = nn.utils.rnn.pack_padded_sequence(src_embed, seq_length, batch_first=True)

            self.console_logger.debug("stsForward decoder src_embed:  %1.3f", torch.sum(src_embed.data))

            enc_h, enc_h_t = self.gru(src_embed, enc_h_0)

            # enc_h, _ = nn.utils.rnn.pad_packed_sequence(enc_h, batch_first=True)

            unsorted_state = enc_h_t.transpose(0, 1).index_select(0, restoration_indices)

            try:
                last_state_index = 2 if self.bidir else 1
            except AttributeError:
                last_state_index = 1

            last_layer_state = unsorted_state[:, -last_state_index:, :]

            self.console_logger.debug("stsForward decoder source:  %1.3f", torch.sum(source.data))
            self.console_logger.debug("stsForward decoder last_layer_state:  %1.3f", torch.sum(last_layer_state.data))

            return enc_h, last_layer_state
