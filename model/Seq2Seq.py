import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from model import Decoder
from model import Encoder
from model import *


class Seq2Seq(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer, embed_dim, hidden_dim, max_len, trg_soi, bi_dir):
        super(Seq2Seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword

        self.encoder = Encoder(src_nword, embed_dim, hidden_dim, num_layer, bi_dir)
        self.decoder = Decoder(trg_nword, embed_dim, hidden_dim, num_layer, bi_dir)

        self.console_logger = logging.getLogger()

    def forward(self, source, src_length=None, target=None, trg_length=None):
        batch_size = source.size(0)

        enc_h, enc_h_t = self.encoder(source, src_length)
        dec_h, dec_h_t = self.decoder(target, trg_length)

        bi_enc_h_t = torch.sum(enc_h_t, dim=1)
        bi_dec_h_t = torch.sum(dec_h_t, dim=1)

        self.console_logger.debug("Seq2Seq bi_enc_h_t:  %1.3f", torch.sum(bi_enc_h_t.data))
        self.console_logger.debug("Seq2Seq bi_dec_h_t:  %1.3f", torch.sum(bi_dec_h_t.data))

        loss = torch.mm(bi_enc_h_t, bi_dec_h_t.transpose(0, 1))
        loss = -1 * loss
        for x in range(0, loss.size()[0]):
            loss[x, x] = - loss[x, x]

        logLoss = torch.log(torch.sigmoid(loss))

        diagonalLoss = 0
        for x in range(0, loss.size()[0]):
            logLoss[x, x] = 40 * logLoss[x, x]
            diagonalLoss += logLoss[x, x]

        logLoss = torch.sum(logLoss)
        logLoss = -1 * logLoss / batch_size
        diagonalLoss = -1 * diagonalLoss / batch_size

        return enc_h_t, enc_h_t, dec_h_t, logLoss, diagonalLoss

    @staticmethod
    def plotInternals(epoch, i, writer, iter_per_epoch, target, bi_dec_h_t, source, bi_enc_h_t):
        info = {
            'decoder source': torch.sum(target.data),
            'encoder source': torch.sum(source.data),
        }
        writer.add_scalars('Encoder_Decoder_Input', info, (epoch * iter_per_epoch) + i + 1)

        info = {
            'decoder last_layer_state': torch.sum(bi_dec_h_t.data),
            'encoder last_layer_state': torch.sum(bi_enc_h_t.data),
        }

        writer.add_scalars('Encoder_Decoder_Output', info, (epoch * iter_per_epoch) + i + 1)

    def logWeightsDataAndGrad(self, epoch, i, writer, iter_per_epoch):
        encoder_weights = torch.sum(self.encoder.gru.all_weights[0][0].data)
        encoder_weights_grad = torch.sum(self.encoder.gru.all_weights[0][0].grad.data)
        decoder_weights = torch.sum(self.decoder.gru.all_weights[0][0].data)
        decoder_weights_grad = torch.sum(self.decoder.gru.all_weights[0][0].grad.data)

        # console
        self.console_logger.debug("encoder : all_weights[0][0].data %1.3f", encoder_weights)
        self.console_logger.debug("encoder : all_weights[0][0].grad %1.8f", encoder_weights_grad)
        self.console_logger.debug("decoder : all_weights[0][0].data %1.3f", decoder_weights)
        self.console_logger.debug("decoder : all_weights[0][0].grad %1.8f", decoder_weights_grad)

        # tensorboard
        info = {
            'encoder_weights': encoder_weights,
            'decoder_weights': decoder_weights,
        }
        writer.add_scalars('Encoder_Decoder_Data', info, (epoch * iter_per_epoch) + i + 1)

        info = {
            'encoder_weights_grad': encoder_weights_grad,
            'decoder_weights_grad': decoder_weights_grad,
        }

        writer.add_scalars('Encoder_Decoder_Grad', info, (epoch * iter_per_epoch) + i + 1)
