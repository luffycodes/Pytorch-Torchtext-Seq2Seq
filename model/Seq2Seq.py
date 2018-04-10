import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from model import Decoder
from model import Encoder
from model import *


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


class Seq2Seq(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer, embed_dim, hidden_dim, max_len, trg_soi, bi_dir):
        super(Seq2Seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword

        self.encoder = Encoder(src_nword, embed_dim, hidden_dim, num_layer, bi_dir)
        self.decoder = Decoder(trg_nword, embed_dim, hidden_dim, num_layer, bi_dir)

        self.console_logger = logging.getLogger()

    def forward(self, source, src_length=None, target=None, trg_length=None, sts=False, batch_sim=None):
        if not sts:
            batch_size = source.size(0)

            enc_h, enc_h_t = self.encoder(source, src_length)
            dec_h, dec_h_t = self.decoder(target, trg_length)

            bi_enc_h_t = torch.sum(enc_h_t, dim=1)
            bi_dec_h_t = torch.sum(dec_h_t, dim=1)

            # bi_enc_h_t = bi_enc_h_t.div(bi_enc_h_t.norm(p=2, dim=1, keepdim=True).expand_as(bi_enc_h_t))
            # bi_dec_h_t = bi_dec_h_t.div(bi_dec_h_t.norm(p=2, dim=1, keepdim=True).expand_as(bi_dec_h_t))

            # self.console_logger.debug("Seq2Seq bi_enc_h_t:  %1.3f", torch.sum(bi_enc_h_t.data))
            # self.console_logger.debug("Seq2Seq bi_dec_h_t:  %1.3f", torch.sum(bi_dec_h_t.data))

            # loss = torch.mm(bi_enc_h_t, bi_dec_h_t.transpose(0, 1))

            loss = pairwise_distances(bi_enc_h_t, bi_dec_h_t)
            nce_loss = 0

            for x in range(0, loss.size()[0]):
                rowLoss = 5
                for y in range(0, loss.size()[0]):
                    rowLoss += - loss[x, y]
                rowLoss += 10 * loss[x, x]
                if rowLoss.data[0] < 0:
                    rowLoss = 0
                nce_loss += rowLoss

            # logLoss = torch.log(torch.sigmoid(loss))

            diagonalLoss = 0
            for x in range(0, loss.size()[0]):
                diagonalLoss += loss[x, x]

            loss = torch.sum(loss)
            loss = -1 * loss / batch_size
            diagonalLoss = -1 * diagonalLoss / batch_size

            return bi_enc_h_t, bi_enc_h_t, bi_dec_h_t, nce_loss, diagonalLoss
        else:
            nn_correlation, enc_h_t, dec_h_t, logLoss, diagonalLoss = self.stsForward(source, src_length, target, trg_length, batch_sim)
            return nn_correlation, enc_h_t, dec_h_t, logLoss, diagonalLoss

    def stsForward(self, source, src_length=None, target=None, trg_length=None, batch_sim=None):
        batch_size = source.size(0)

        enc_h, enc_h_t = self.encoder(source, src_length, sts=True, sort=True)
        dec_h, dec_h_t = self.encoder(target, trg_length, sts=True, sort=True)

        bi_enc_h_t = torch.sum(enc_h_t, dim=1)
        bi_dec_h_t = torch.sum(dec_h_t, dim=1)

        bi_enc_h_t = bi_enc_h_t.div(bi_enc_h_t.norm(p=2, dim=1, keepdim=True).expand_as(bi_enc_h_t))
        bi_dec_h_t = bi_dec_h_t.div(bi_dec_h_t.norm(p=2, dim=1, keepdim=True).expand_as(bi_dec_h_t))

        # self.console_logger.debug("Seq2Seq bi_enc_h_t:  %1.3f", torch.sum(bi_enc_h_t.data))
        # self.console_logger.debug("Seq2Seq bi_dec_h_t:  %1.3f", torch.sum(bi_dec_h_t.data))

        loss = torch.mm(bi_enc_h_t, bi_dec_h_t.transpose(0, 1))
        loss = -1 * loss
        for x in range(0, loss.size()[0]):
            loss[x, x] = - loss[x, x]

        sigmoidLoss = torch.sigmoid(loss)
        nn_correlation = []
        nn_correlation_loss = 0
        for x in range(0, loss.size()[0]):
            nn_correlation.append(sigmoidLoss[x, x].data[0] * 5)
            correlation_diff = sigmoidLoss[x, x] * 500 - batch_sim[x].data[0]
            nn_correlation_loss = correlation_diff * correlation_diff

        self.console_logger.debug('bi_enc_h_t_0 {0}'.format(bi_enc_h_t.data[0].cpu().numpy()))
        self.console_logger.debug('bi_dec_h_t_0 {0}'.format(bi_dec_h_t.data[0].cpu().numpy()))
        self.console_logger.debug('loss h_t_0 {0}'.format(sigmoidLoss[0][0].data[0]))
        self.console_logger.debug('bi_enc_h_t_1 {0}'.format(bi_enc_h_t.data[1].cpu().numpy()))
        self.console_logger.debug('bi_dec_h_t_1 {0}'.format(bi_dec_h_t.data[1].cpu().numpy()))
        self.console_logger.debug('loss h_t_1 {0}'.format(sigmoidLoss[1][1].data[0]))

        logLoss = torch.log(sigmoidLoss)
        diagonalLoss = 0
        for x in range(0, loss.size()[0]):
            logLoss[x, x] = 10 * logLoss[x, x]
            diagonalLoss += logLoss[x, x]

        logLoss = torch.sum(logLoss)
        logLoss = -1 * logLoss / batch_size
        diagonalLoss = -1 * diagonalLoss / batch_size
        nn_correlation_loss = nn_correlation_loss / batch_size

        return nn_correlation, nn_correlation_loss, dec_h_t, logLoss, diagonalLoss

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
