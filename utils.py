import random

import torch
from torch.autograd import Variable


def tensor2np(tensor):
    return tensor.data.cpu().numpy()


def randomChoice(batch_size):
    return random.randint(0, batch_size - 1)


def sort_batch_by_length(tensor: torch.autograd.Variable, sequence_lengths: torch.autograd.Variable):
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

    if torch.cuda.is_available():
        sorted_sequence_lengths = sorted_sequence_lengths.cuda()
        permutation_index = permutation_index.cuda()

    sorted_tensor = tensor.index_select(0, permutation_index)

    if torch.cuda.is_available():
        sorted_tensor = sorted_tensor.cuda()

    # This is ugly, but required - we are creating a new variable at runtime, so we
    # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
    # refilling one of the inputs to the function.
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    index_range = Variable(index_range.long())
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)

    if torch.cuda.is_available():
        restoration_indices = restoration_indices.cuda()

    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Borrowed from ImageNet training in PyTorch project
    https://github.com/pytorch/examples/tree/master/imagenet
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
