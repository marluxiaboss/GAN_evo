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


class BERT_sentiment(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(BERT_sentiment, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'dpgan_d'
        self.sentiment = pipeline(task='sentiment-analysis')
        self.bpe = get_encoder()

    def score_token(self, score, label):
        if label == 'POSITIVE':
            return score
        else:
            return 1.0 - score

    def getReward(self, samples, training_bin, pos_or_neg_sample=None):
        """
        Get word-level reward and sentence-level reward of one sample.
        """

        """
        word_reward = F.nll_loss(pred, target.view(-1), reduction='none').view(batch_size, -1)
        sentence_reward = torch.mean(word_reward, dim=-1, keepdim=True)
        """
        sample = self.bpe.decode(samples.tolist())

        sentiments = self.sentiment(sample)

        print("SAMPLES")
        print(sample)
        print("SENTENCE_reward")
        print(sentiments)

        label_map = {'NEGATIVE': 0.0, 'POSITIVE': 1.0}
        sentence_rewards = torch.tensor([self.score_token(sentiment['score'], sentiment['label']) for sentiment in
                                         sentiments], requires_grad=True)
        sentence_sentiment = sentence_rewards.view(1, len(sentence_rewards))
        # maybe better to give rewards for only this length and not cfg.max_seqlen
        #word_rewards = [sentence_rewards for i in range(len(samples[0]))]
        for sentiment in sentiments:
            training_bin[int(label_map[sentiment['label']]) - 1] += 1
        print("SENTENCE_SENTIMENT")
        print(sentence_sentiment)
        return sentence_sentiment
