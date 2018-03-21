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
        sorted_inputs, sorted_seq_len, restoration_indices, _ = self.sort_batch_by_length(target, Variable(torch.FloatTensor(trg_length)).cuda())

        src_embed = self.embedding(sorted_inputs)

        if hidden is None:
            h_size = (self.num_layers *2, batch_size, self.hidden_dim)
            enc_h_0 = Variable(src_embed.data.new(*h_size).zero_(), requires_grad=False)

        seq_length = [int(x) for x in sorted_seq_len.data.numpy().tolist()]
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

    def sort_batch_by_length(self, tensor: torch.autograd.Variable, sequence_lengths: torch.autograd.Variable):
        print("inside of sort_batch_by_length")

        """
        Sort a batch first tensor by some specified lengths.

        Parameters
        ----------
        tensor : Variable(torch.FloatTensor), required.
            A batch first Pytorch tensor.
        sequence_lengths : Variable(torch.LongTensor), required.
            A tensor representing the lengths of some dimension of the tensor which
            we want to sort by.

        Returns
        -------
        sorted_tensor : Variable(torch.FloatTensor)
            The original tensor sorted along the batch dimension with respect to sequence_lengths.
        sorted_sequence_lengths : Variable(torch.LongTensor)
            The original sequence_lengths sorted by decreasing size.
        restoration_indices : Variable(torch.LongTensor)
            Indices into the sorted_tensor such that
            ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
        permuation_index : Variable(torch.LongTensor)
            The indices used to sort the tensor. This is useful if you want to sort many
            tensors using the same ordering.
        """

        if not isinstance(tensor, Variable) or not isinstance(sequence_lengths, Variable):
            raise Exception("Both the tensor and sequence lengths must be torch.autograd.Variables.")

        sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
        sorted_tensor = tensor.index_select(0, permutation_index)

        # This is ugly, but required - we are creating a new variable at runtime, so we
        # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
        # refilling one of the inputs to the function.
        index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
        # This is the equivalent of zipping with index, sorting by the original
        # sequence lengths and returning the now sorted indices.
        index_range = Variable(index_range.long()).cuda()
        _, reverse_mapping = permutation_index.sort(0, descending=False)
        restoration_indices = index_range.index_select(0, reverse_mapping)

        print("out of sort_batch_by_length")

        return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index
