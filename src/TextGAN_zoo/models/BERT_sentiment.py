# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : SeqGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn.functional as F

import config as cfg
from models.generator import LSTMGenerator
from utils.bp_encoder import get_encoder
from utils.data_loader import GenDataIter
from transformers import pipeline


class DPGAN_D(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(DPGAN_D, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'dpgan_d'
        self.sentiment = pipeline(task='sentiment-analysis',
                                  model='nlptown/bert-base-multilingual-uncased-sentiment',
                                  tokenizer='nlptown/bert-base-multilingual-uncased-sentiment')
        self.bpe = get_encoder()
    def getReward(self, samples, pos_or_neg_sample=None):
        """
        Get word-level reward and sentence-level reward of samples.
        """


        """
        word_reward = F.nll_loss(pred, target.view(-1), reduction='none').view(batch_size, -1)
        sentence_reward = torch.mean(word_reward, dim=-1, keepdim=True)
        """
        word_reward = None
        samples = samples.tolist()
        samples = [[self.bpe.decode(sample)] for sample in samples]
        sentence_reward = self.sentiment(samples)

        return word_reward, sentence_reward