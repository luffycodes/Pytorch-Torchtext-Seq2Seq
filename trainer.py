import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *

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
    def __init__(self, train_loader, val_loader, vocabs, args):

        # Language setting
        self.max_len = args.max_len

        # Data Loader
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Path
        self.data_path = args.data_path
        self.log_path = os.path.join('./logs/' + args.log)

        if not os.path.exists(self.log_path): os.makedirs(self.log_path)

        # Hyper-parameters
        self.lr = args.lr
        self.grad_clip = args.grad_clip
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.num_layer = args.num_layer

        # Training setting
        self.batch_size = args.batch_size
        self.num_epoch = args.num_epoch
        self.iter_per_epoch = len(train_loader)
        self.best_loss = .0

        # Log
        self.tf_log = Logger(self.log_path)
        self.train_loss = AverageMeter()

        self.console_logger = logging.getLogger()
        self.console_logger.setLevel(logging.DEBUG)
        consoleHandler = logging.StreamHandler()
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        consoleHandler.setFormatter(logFormatter)
        self.console_logger.addHandler(consoleHandler)
        self.console_logger.propagate = False

        self.build_model(vocabs)

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
                             self.max_len, self.trg_soi)

        # set the criterion and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.8)

        if torch.cuda.is_available():
            self.model.cuda()

        print(self.model)
        print(self.criterion)
        print(self.optimizer)

    def train(self):
        for epoch in range(self.num_epoch):
            # self.scheduler.step()
            self.train_loss.reset()
            start_time = time.time()

            for i, batch in enumerate(self.train_loader):
                self.console_logger.debug("epoch:%d, i:%d", epoch, i)
                self.model.train()

                src_input = batch.src[0]
                src_length = batch.src[1]
                trg_input = batch.trg[0]
                trg_length = batch.trg[1]

                batch_size, trg_len = trg_input.size(0), trg_input.size(1)

                _, enc_h_t, dec_h_t, loss = self.model(src_input, src_length.tolist(), trg_input,
                                                                   trg_length.tolist())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                self.train_loss.update(loss.data[0] / (batch_size * batch_size), 1)

                if i % 3 == 0:
                    self.log_train_result(epoch, i, start_time)
                    self.eval(epoch, i)

                    # Logging tensorboard
                    info = {
                        'epoch': epoch,
                        'train_iter': i,
                        'train_loss': self.train_loss.avg,
                    }
                    for tag, value in info.items():
                        self.tf_log.scalar_summary(tag, value, (epoch * self.iter_per_epoch) + i + 1)

                    # reset for next 100 batch
                    self.train_loss.reset()
                    start_time = time.time()

            self.log_train_result(epoch, i, start_time)
            self.eval(epoch, i)

    def eval(self, epoch, train_iter):
        self.model.eval()

        val_loss = AverageMeter()
        start_time = time.time()

        for i, batch in enumerate(tqdm(self.val_loader)):
            src_input = batch.src[0]
            src_length = batch.src[1]
            trg_input = batch.trg[0]
            trg_length = batch.trg[1]

            batch_size, trg_len = trg_input.size(0), trg_input.size(1)

            _, enc_h_t, dec_h_t, loss = self.model(src_input, src_length.tolist(), trg_input,
                                                   trg_length.tolist())

            val_loss.update(loss / (batch_size * batch_size), 1)

        self.log_valid_result(epoch, train_iter, val_loss.avg, start_time)

        # Save model if bleu score is higher than the best 
        if self.best_loss < val_loss.avg.data[0]:
            self.best_loss = val_loss.avg.data[0]
            checkpoint = {
                'model': self.model,
                'epoch': epoch
            }
            # torch.save(checkpoint, self.log_path + '/Model_e%d_i%d_%.3f.pt' % (epoch, train_iter, val_loss.avg))

        # Logging tensorboard
        info = {
            'epoch': epoch,
            'train_iter': train_iter,
            'train_loss': self.train_loss.avg,
            'val_loss': val_loss.avg
        }

        for tag, value in info.items():
            self.tf_log.scalar_summary(tag, value, (epoch * self.iter_per_epoch) + train_iter + 1)

    def log_train_result(self, epoch, train_iter, start_time):
        message = "Training :: Train epoch: %d  iter: %d  train loss: %1.3f elapsed: %1.3f " % (
            epoch, train_iter, self.train_loss.avg, time.time() - start_time)

        self.console_logger.debug(message)

    def log_valid_result(self, epoch, train_iter, val_loss, start_time):
        message = "Validation :: Train epoch: %d  iter: %d  val loss score: %1.3f train loss: %1.3f elapsed: " \
                  "%1.3f " % (
                      epoch, train_iter, val_loss, self.train_loss.avg, time.time() - start_time)

        self.console_logger.debug(message)
