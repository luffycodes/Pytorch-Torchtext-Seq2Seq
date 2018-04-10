import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
from tensorboardX import SummaryWriter

import numpy as np
import math
import time
import os
import logging

from logger import Logger
from tqdm import tqdm

from prepro import *
from utils import *
from model.Seq2Seq import Seq2Seq
from bleu import *


class Trainer(object):
    def __init__(self, train_loader, val_loader, sts_loader, vocabs, correlation, args):

        # Language setting
        self.max_len = args.max_len

        # Data Loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.sts_loader = sts_loader

        # Path
        self.data_path = args.data_path
        self.log_path = os.path.join(args.log_path + args.log)

        if not os.path.exists(self.log_path): os.makedirs(self.log_path)

        # Hyper-parameters
        self.lr = args.lr
        self.grad_clip = args.grad_clip
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer
        self.bi_dir = args.bi_dir

        # Training setting
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        self.iter_per_epoch = len(train_loader)
        self.best_loss = .0

        # Log
        self.tf_log = SummaryWriter(self.log_path)
        self.train_loss = AverageMeter()
        self.diagonal_loss = AverageMeter()

        self.console_logger = logging.getLogger()

        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_num)

        self.build_model(vocabs)

        # STS
        self.correlation = correlation

    def build_model(self, vocabs):
        # build dictionaries
        self.src_vocab = vocabs['src_vocab']
        self.trg_vocab = vocabs['trg_vocab']
        self.src_inv_vocab = vocabs['src_inv_vocab']
        self.trg_inv_vocab = vocabs['trg_inv_vocab']
        self.trg_soi = self.trg_vocab[SOS_WORD]

        self.src_nword = len(self.src_vocab)
        self.trg_nword = len(self.trg_vocab)

        # build the model
        self.model = Seq2Seq(self.src_nword, self.trg_nword, self.num_layer, self.embed_dim, self.hidden_dim,
                             self.max_len, self.trg_soi, self.bi_dir)

        # set the criterion and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.8)

        if torch.cuda.is_available():
            self.model.cuda()

        self.console_logger.debug(self.model)
        self.console_logger.debug(self.criterion)
        self.console_logger.debug(self.optimizer)

    def train(self):
        for epoch in range(self.num_epoch):
            # self.scheduler.step()
            self.train_loss.reset()
            self.diagonal_loss.reset()
            start_time = time.time()

            for i, batch in enumerate(self.train_loader):
                self.model.train()

                src_input = batch.src[0]
                src_length = batch.src[1]
                trg_input = batch.trg[0]
                trg_length = batch.trg[1]

                batch_size, trg_len = trg_input.size(0), trg_input.size(1)

                _, enc_h_t, dec_h_t, loss, diagonalLoss = self.model(src_input, src_length.tolist(), trg_input,
                                                                     trg_length.tolist())
                if i % 10000 == 0 and i != 0:
                    self.console_logger.debug('train_enc_h_t_0 {0} {1} {2}'.format(epoch, i, enc_h_t.data[0].cpu().numpy()))
                    self.console_logger.debug('train_dec_h_t_0 {0} {1} {2}'.format(epoch, i, dec_h_t.data[0].cpu().numpy()))
                    self.console_logger.debug('train_enc_h_t_1 {0} {1} {2}'.format(epoch, i, enc_h_t.data[1].cpu().numpy()))
                    self.console_logger.debug('train_dec_h_t_1 {0} {1} {2}'.format(epoch, i, dec_h_t.data[1].cpu().numpy()))

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                self.train_loss.update(loss.data[0], 1)
                self.diagonal_loss.update(diagonalLoss.data[0], 1)

                if i % 1000 == 0 and i != 0 and epoch > 0:
                    self.console_logger.debug("epoch:%d, i:%d, iter_per_epoch:%d", epoch, i, self.iter_per_epoch)
                    self.log_train_result(epoch, i, start_time)
                    self.eval(epoch, i)

                    # Logging tensorboard
                    info = {
                        'train_loss': self.train_loss.avg,
                        'diagonal_loss': self.diagonal_loss.avg,
                        'negative_sample': self.train_loss.avg - self.diagonal_loss.avg,
                    }
                    self.tf_log.add_scalars('Training loss', info, (epoch * self.iter_per_epoch) + i + 1)
                    self.model.plotInternals(epoch, i, self.tf_log, self.iter_per_epoch, trg_input, dec_h_t, src_input,
                                             enc_h_t)
                    self.model.logWeightsDataAndGrad(epoch, i, self.tf_log, self.iter_per_epoch)

                    # reset for next 100 batch
                    self.train_loss.reset()
                    self.diagonal_loss.reset()
                    start_time = time.time()

                if i % 1000 == 0 and i != 0:
                    pass
                    # self.stsEval(epoch, i)

            self.log_train_result(epoch, i, start_time)
            self.eval(epoch, i)

    def stsEval(self, epoch, train_iter):
        self.console_logger.debug("entering sts code")
        self.model.train()

        sts_loss = AverageMeter()
        sts_diagonal_loss = AverageMeter()
        nn_correlation_loss_meter = AverageMeter()
        start_time = time.time()
        nn_correlation = []

        for i, batch in enumerate(self.sts_loader):
            src_input = batch.src[0]
            src_length = batch.src[1]
            trg_input = batch.trg[0]
            trg_length = batch.trg[1]

            batch_size, trg_len = trg_input.size(0), trg_input.size(1)

            nn_correlation_batch, nn_correlation_loss, dec_h_t, loss, diagonalLoss = self.model(src_input,
                                                                                    src_length.tolist(),
                                                                                    trg_input,
                                                                                    trg_length.tolist(), sts=True,
                                                                                    batch_sim=batch.sim)

            self.optimizer.zero_grad()
            nn_correlation_loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            nn_correlation_loss_meter.update(nn_correlation_loss.data[0], 1)
            sts_loss.update(loss.data[0], 1)
            sts_diagonal_loss.update(diagonalLoss.data[0], 1)

            for j in range(len(nn_correlation_batch)):
                nn_correlation.append(nn_correlation_batch[j])

        for k in range(len(nn_correlation)):
            if k % 10 == 0:
                self.console_logger.debug('sts correlation %d, %d, %d, %1.3f, %1.3f', epoch, train_iter, k, nn_correlation[k], self.correlation[k])

        correlation = pearson_correlation(self.correlation, nn_correlation)

        self.log_sts_result(epoch, train_iter, sts_loss.avg, start_time)

        # Logging tensorboard
        info = {
            'sts_loss': sts_loss.avg,
            'sts_diagonal_loss': sts_diagonal_loss.avg,
            'sts_negative_sample': sts_loss.avg - sts_diagonal_loss.avg,
            'sts_correlation': correlation,
            'nn_correlation_loss': nn_correlation_loss_meter.avg,
        }

        self.tf_log.add_scalars('sts_summary_loss', info, (epoch * self.iter_per_epoch) + train_iter + 1)

        self.console_logger.debug("exiting sts code")

    def eval(self, epoch, train_iter):
        self.console_logger.debug("entering validation code")
        self.model.eval()

        val_loss = AverageMeter()
        val_diagonal_loss = AverageMeter()
        start_time = time.time()

        for i, batch in enumerate(tqdm(self.val_loader)):
            src_input = batch.src[0]
            src_length = batch.src[1]
            trg_input = batch.trg[0]
            trg_length = batch.trg[1]

            batch_size, trg_len = trg_input.size(0), trg_input.size(1)

            _, enc_h_t, dec_h_t, loss, diagonalLoss = self.model(src_input, src_length.tolist(), trg_input,
                                                                 trg_length.tolist())

            similarity = torch.sigmoid(torch.mm(enc_h_t, enc_h_t.transpose(0, 1)))
            for x in range(0, similarity.size()[0]):
                self.console_logger.debug('src trg epoch iter sim  %d %d %d, %d, %1.3f', batch.key[0], batch.key[x], epoch, train_iter, similarity[0, x])
                self.console_logger.debug('valid_enc_h_t_0 {0}, {1}, {2}'.format(epoch, train_iter, enc_h_t.data[0].cpu().numpy()))
                self.console_logger.debug('valid_enc_h_t_0 {0}, {1}, {2}'.format(epoch, train_iter, enc_h_t.data[x].cpu().numpy()))

            val_loss.update(loss.data[0], 1)
            val_diagonal_loss.update(diagonalLoss.data[0], 1)

        self.log_valid_result(epoch, train_iter, val_loss.avg, start_time)

        # Save model if bleu score is higher than the best 
        if self.best_loss < val_loss.avg:
            self.best_loss = val_loss.avg
            checkpoint = {
                'model': self.model,
                'epoch': epoch
            }
            # torch.save(checkpoint, self.log_path + '/Model_e%d_i%d_%.3f.pt' % (epoch, train_iter, val_loss.avg))

        # Logging tensorboard
        info = {
            'val_loss': val_loss.avg,
            'val_diagonal_loss': val_diagonal_loss.avg,
            'val_negative_sample': val_loss.avg - val_diagonal_loss.avg,
        }

        self.tf_log.add_scalars('Validation loss', info, (epoch * self.iter_per_epoch) + train_iter + 1)

        self.console_logger.debug("exiting validation code")

    def log_train_result(self, epoch, train_iter, start_time):
        message = "Training $#$ Epoch:%d $#$ iter: %d $#$ training_loss: %1.3f $#$ elapsed: %1.3f " % (
            epoch, train_iter, self.train_loss.avg, time.time() - start_time)

        self.console_logger.debug(message)

    def log_valid_result(self, epoch, train_iter, val_loss, start_time):
        message = "Validation $#$ Epoch: %d $#$ iter: %d $#$ validation_loss: %1.3f $#$ training_loss: %1.3f elapsed: " \
                  "%1.3f " % (
                      epoch, train_iter, val_loss, self.train_loss.avg, time.time() - start_time)

        self.console_logger.debug(message)

    def log_sts_result(self, epoch, train_iter, val_loss, start_time):
        message = "STS $#$ Epoch: %d $#$ iter: %d $#$ validation_loss: %1.3f $#$ training_loss: %1.3f elapsed: " \
                  "%1.3f " % (
                      epoch, train_iter, val_loss, self.train_loss.avg, time.time() - start_time)

        self.console_logger.debug(message)
