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
from utils.data_loader import GenDataIter
from utils.text_process import load_dict


class DPGAN_D(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(DPGAN_D, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'dpgan_d'

    def getReward(self, samples):
        """
        Get word-level reward and sentence-level reward of samples.
        """
        batch_size, _ = samples.size()
        inp, target = GenDataIter.prepare(samples, cfg.CUDA)

        hidden = self.init_hidden(batch_size)
        pred = self.forward(inp, hidden)

        word_reward = F.nll_loss(pred, target.view(-1), reduction='none').view(batch_size, -1)
        print("PRED")
        onebatch_pred = pred[:cfg.max_seq_len, :]
        for i in range(10):
            one_batch_pred_toki = onebatch_pred[i]
            prob_top_pred, sample_top_pred = torch.topk(one_batch_pred_toki, 3)
            tokens_pred = [self.idx2word_dict[str(i)] for i in sample_top_pred.tolist()]
            print(str(i) + "     : ", end='')
            print(tokens_pred)
        print("TARGET")
        onebatch_targ = target[0]
        tokens_targ = [self.idx2word_dict[str(i)] for i in onebatch_targ.tolist()]
        print(tokens_targ)
        print("WORD_REWARD")
        onebatch_reward = word_reward[0]
        print(onebatch_reward)
        sentence_reward = torch.mean(word_reward, dim=-1, keepdim=True)

        return word_reward, sentence_reward
