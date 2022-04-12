# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : dpgan_instructor.py
# @Time         : Created at 2019/12/21
# @Blog         : http://zhiweil.ml/
# @Description  :
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.optim as optim

import config as cfg
from instructor.real_data.instructor import BasicInstructor, SelfAttentionInstructor
from models.DPGAN_D import DPGAN_D
from models.GPT_2 import GPT_2
from transformers import GPT2Model, GPT2Tokenizer

from utils import helpers
from utils.data_loader import GenDataIter


class GPT_BERT_DPGAN(SelfAttentionInstructor):
    def __init__(self, opt):
        super(GPT_BERT_DPGAN, self).__init__(opt)

        # generator, discriminator
        self.gen = GPT_2()
        self.dis = DPGAN_D(cfg.gen_embed_dim, cfg.gen_hidden_dim, self.gen.config.vocab_size, cfg.max_seq_len,
                           cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

        # Load weights from huggingface GPT_2 transformer class
        pretrained_model = GPT2Model.from_pretrained("gpt2")
        self.gen = helpers.load_weight(self.gen, pretrained_model.state_dict())

        # Tokenizer for the pretrained gpt2
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def init_model(self):
        """
        Overwrites the init_model() in instructor.py
        """
        if cfg.dis_pretrain:
            self.log.info(
                'Load pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))
            self.dis.load_state_dict(torch.load(cfg.pretrained_dis_path, map_location='cuda:{}'.format(cfg.device)))

        if cfg.CUDA:
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()

    def _run(self):
        # # ===TRAIN DISCRIMINATOR====
        if not cfg.dis_pretrain:
            self.log.info('Starting Discriminator Training...')
            self.train_discriminator(cfg.d_step, cfg.d_epoch, 'MLE')
            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)
                print('Save pre-trained discriminator: {}'.format(cfg.pretrained_dis_path))

        # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        self.log.info('Initial generator: %s' % (self.cal_metrics(fmt_str=True)))

        for adv_epoch in range(cfg.ADV_train_epoch):
            self.log.info('-----\nADV EPOCH %d\n-----' % adv_epoch)
            self.sig.update()
            if self.sig.adv_sig:
                self.adv_train_generator(cfg.ADV_g_step)  # Generator
                self.train_discriminator(cfg.ADV_d_step, cfg.ADV_d_epoch, 'ADV')  # Discriminator

                if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                    if cfg.if_save and not cfg.if_test:
                        self._save('ADV', adv_epoch)
            else:
                self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
                break

    def _test(self):
        print('>>> Begin test...')

        self._run()
        pass


    def adv_train_generator(self, g_step):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        discount_rate = 1
        total_g_loss = 0
        dis_count_list = [discount_rate ** i for i in range(cfg.max_seq_len)]
        dis_count_matrix = torch.Tensor(dis_count_list).unsqueeze(0).repeat(cfg.batch_size, 1)
        if cfg.CUDA:
            dis_count_matrix = dis_count_matrix.cuda()

        for step in range(g_step):
            inp = self.train_data.random_batch()['input']
            if cfg.CUDA:
                inp = inp.cuda()

            gen_sample, gen_sample_log_prob = self.gen.sample_teacher_forcing(inp)
            word_reward, sentence_reward = self.dis.getReward(gen_sample, pos_or_neg_sample = "NEG")
            sentence_reward = sentence_reward.repeat(1, cfg.max_seq_len)
            reward_matrix = sentence_reward * word_reward * dis_count_matrix
            for i in range(cfg.max_seq_len):
                reward_matrix[:, i] = reward_matrix[:, i:].sum(dim=-1)

            adv_loss = torch.sum(gen_sample_log_prob * reward_matrix)

            self.optimize(self.gen_adv_opt, adv_loss, self.gen)
            total_g_loss += adv_loss.item()

        # ===Test===
        self.log.info(
            '[ADV-GEN]: g_loss = %.4f, %s' % (total_g_loss / (g_step * cfg.batch_size), self.cal_metrics(fmt_str=True)))

    def train_discriminator(self, d_step, d_epoch, phase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        # prepare loader for validate
        for step in range(d_step):
            # prepare loader for training
            pos_samples = self.train_data.target[:100,:]
            neg_samples = self.gen.sample_sequence(cfg.max_seq_len - 1, start_token=cfg.start_letter,
                                                   batch_size=pos_samples.size(0), temperature=0.7, top_k=40)

            pos_reward, neg_reward = 0, 0
            for epoch in range(d_epoch):
                # ===Train===
                pos_reward, neg_reward = self.train_dis_epoch(self.dis, pos_samples, neg_samples, self.dis_opt)

            # ===Test===
            self.log.info('[%s-DIS] d_step %d: pos_reward = %.4f, neg_reward = %.4f,' % (
                phase, step, pos_reward, neg_reward))

            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)

    def eval_dis(self, model, pos_val, neg_val):
        _, pos_reward = model.getReward(pos_val, pos_or_neg_sample="POS")
        _, neg_reward = model.getReward(neg_val, pos_or_neg_sample="NEG")
        return torch.mean(pos_reward), torch.mean(neg_reward)

    def train_dis_epoch(self, model, pos_samples, neg_samples, optimizer):
        pos_reward, neg_reward = 0, 0
        num_samples = pos_samples.size(0)
        num_batch = num_samples // cfg.batch_size
        for i in range(num_batch):
            pos_sample = pos_samples[i * cfg.batch_size: (i + 1) * cfg.batch_size]
            neg_sample = neg_samples[i * cfg.batch_size: (i + 1) * cfg.batch_size]

            _, pos_reward = model.getReward(pos_sample, pos_or_neg_sample="POS")
            _, neg_reward = model.getReward(neg_sample, pos_or_neg_sample="NEG")

            loss = -torch.mean(pos_reward) + torch.mean(neg_reward)
            self.optimize(optimizer, loss, model)
        return pos_reward.mean().item(), neg_reward.mean().item()


    def cal_metrics(self, fmt_str=False):
        """
        Overwrites cal_metrics from BasicInstructor, because we need to use a specific
        tokenizer for the pretrained gpt2
        """
        with torch.no_grad():
            # Prepare data for evaluation
            eval_samples = self.gen.sample_sequence(cfg.max_seq_len - 1, start_token=cfg.start_letter,
                                                    batch_size=cfg.samples_num, temperature=0.7, top_k=40,
                                                    sample_pos2=True)
            gen_data = GenDataIter(eval_samples)
            #gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
            gen_tokens = self.tokenizer.decode(eval_samples)
            # gen_tokens_s = tensor_to_tokens(self.gen.sample_sequence(cfg.max_seq_len - 1, start_token=cfg.start_letter,
            #                                        batch_size=200, temperature=0.7, top_k=40), self.idx2word_dict)

            # Reset metrics
            self.bleu.reset(test_text=gen_tokens, real_text=self.test_data.tokens)
            self.nll_gen.reset(self.gen, self.train_data.loader)
            self.nll_div.reset(self.gen, gen_data.loader)
            # self.self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
            self.ppl.reset(gen_tokens)

        if fmt_str:
            return ', '.join(['%s = %s' % (metric.get_name(), metric.get_score()) for metric in self.all_metrics])
        else:
            return [metric.get_score() for metric in self.all_metrics]

