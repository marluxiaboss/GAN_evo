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
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax


class BERT_sentiment(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(BERT_sentiment, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'roberta'
        self.bpe = get_encoder()
        self.model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.sentiment= AutoModelForSequenceClassification.from_pretrained(MODEL)
    def score_token(self, score, label):
        if label == 'POSITIVE':
            return score
        else:
            return 1.0 - score
    def getReward(self, samples, training_bin, one_sample=False, pos_or_neg_sample=None):
        """
        Get word-level reward and sentence-level reward of samples.
        """


        """
        word_reward = F.nll_loss(pred, target.view(-1), reduction='none').view(batch_size, -1)
        sentence_reward = torch.mean(word_reward, dim=-1, keepdim=True)
        """

        if one_sample:
            samples = self.bpe.decode(samples.tolist())
        else:
            samples = samples.tolist()
            samples = [[self.bpe.decode(sample)] for sample in samples]
        # TODO: would be better to use the input as a tensor to be
        # able to use the gpu
        encoded_input = self.tokenizer(samples, return_tensors='pt')
        output = self.sentiment(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = self.config.id2label[ranking[i]]
            s = scores[ranking[i]]
            print(f"{i + 1}) {l} {np.round(float(s), 4)}")

        sentence_sentiment = None
        """
        print("SAMPLES_BERT")
        print(samples)
        print("SENTIMENTS")
        print(sentiments)
        """

        

        """
        label_map = {'NEGATIVE': 0.0, 'POSITIVE': 1.0}
        sentence_sentiment = torch.tensor([self.score_token(sentiment['score'], sentiment['label']) for sentiment in
                                         sentiments], requires_grad=False)
        #sentence_sentiment = sentence_rewards.view(1, len(sentence_rewards))
        # maybe better to give rewards for only this length and not cfg.max_seqlen
        #word_rewards = [sentence_rewards for i in range(len(samples[0]))]
        for sentiment in sentiments:
            training_bin[int(label_map[sentiment['label']])] += 1
        #print("SENTENCE_SENTIMENT")
        #print(sentence_sentiment)
        """
        return sentence_sentiment

