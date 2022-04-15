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
from transformers import GPT2Model, GPT2Tokenizer

import config as cfg
from instructor.real_data.instructor import SelfAttentionInstructor
from models.BERT_sentiment import BERT_sentiment
from models.GPT_2 import GPT_2
from utils import helpers
from utils.bp_encoder import get_encoder
from utils.data_loader import GenDataIter
from utils.text_process import write_tokens


class GPT_BERT_DPGAN(SelfAttentionInstructor):
    def __init__(self, opt):
        super(GPT_BERT_DPGAN, self).__init__(opt)

        # generator, discriminator
        self.gen = GPT_2()
        self.dis = BERT_sentiment(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
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
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.bpe = get_encoder()

    def init_model(self):
        """
        Overwrites the init_model() in instructor.py
        """
        if cfg.CUDA:
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()

    def _run(self):
        # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        self.log.info('Initial generator: %s' % (self.cal_metrics(fmt_str=True)))

        for adv_epoch in range(cfg.ADV_train_epoch):
            self.log.info('-----\nADV EPOCH %d\n-----' % adv_epoch)
            self.sig.update()
            if self.sig.adv_sig:
                self.adv_train_generator(cfg.ADV_g_step)  # Generator

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
            word_reward, sentence_reward = self.dis.getReward(gen_sample)
            if word_reward is not None:
                sentence_reward = sentence_reward.repeat(1, cfg.max_seq_len)
                reward_matrix = sentence_reward * word_reward * dis_count_matrix
            else:
                reward_matrix = sentence_reward
            for i in range(cfg.max_seq_len):
                reward_matrix[:, i] = reward_matrix[:, i:].sum(dim=-1)

            adv_loss = torch.sum(gen_sample_log_prob * reward_matrix)

            self.optimize(self.gen_adv_opt, adv_loss, self.gen)
            total_g_loss += adv_loss.item()

        # ===Test===
        self.log.info(
            '[ADV-GEN]: g_loss = %.4f, %s' % (total_g_loss / (g_step * cfg.batch_size), self.cal_metrics(fmt_str=True)))

    def eval_dis(self, model, pos_val, neg_val):
        _, pos_reward = model.getReward(pos_val)
        _, neg_reward = model.getReward(neg_val)
        return torch.mean(pos_reward), torch.mean(neg_reward)

    def cal_metrics(self, fmt_str=False):
        """
        Overwrites cal_metrics from BasicInstructor, because we need to use a specific
        tokenizer for the pretrained gpt2
        """
        with torch.no_grad():
            # Prepare data for evaluation
            eval_samples = self.gen.sample_sequence(cfg.max_seq_len - 1, start_token=cfg.start_letter,
                                                    batch_size=cfg.samples_num, temperature=0.7, top_k=40,
                                                    sample_pos2=False)
            gen_data = GenDataIter(eval_samples)
            # gen_tokens = tensor_to_tokens(eval_samples, self.idx2word_dict)
            eval_samples = eval_samples.tolist()
            gen_tokens = [[self.bpe.decode(eval_sample)] for eval_sample in eval_samples]
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

    def _save(self, phase, epoch):
        """Overwrites _save in instructor to add gpt2 tokenizer"""
        if phase != 'ADV':
            torch.save(self.gen.state_dict(), cfg.save_model_root + 'gen_{}_{:05d}.pt'.format(phase, epoch))
        save_sample_path = cfg.save_samples_root + 'samples_{}_{:05d}.txt'.format(phase, epoch)
        samples = self.gen.sample_sequence(cfg.max_seq_len - 1, start_token=cfg.start_letter,
                                           batch_size=50, temperature=0.7, top_k=40)
        samples = samples.tolist()
        # samples = [[self.tokenizer.decode(sample)] for sample in samples]
        samples = [[self.bpe.decode(sample)] for sample in samples]
        write_tokens(save_sample_path, samples)
