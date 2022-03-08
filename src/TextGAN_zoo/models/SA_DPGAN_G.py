import torch
import torch.nn.functional as F

from models.generator import TransformerGenerator
import config as cfg


class SA_DPGAN_G(TransformerGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, num_heads=4, nlayers=3, dropout=0.5, gpu=False):
        super(SA_DPGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, num_heads, nlayers, dropout, gpu)
        self.name = 'sa_dpgan_g'

    def sample_teacher_forcing(self, inp):
        """
        Generating samples from the real data via teacher forcing
        :param inp: batch_size * seq_len
        :param target: batch_size * seq_len
        :return
            samples: batch_size * seq_len
            log_prob: batch_size * seq_len  (log probabilities)
        """
        batch_size, _ = inp.size()
        inp = inp.transpose(1, 0)       # [max_seq_len, batch_size]
        dummy_tgt = torch.ones(self.max_seq_len, batch_size, dtype=torch.int)
        if self.gpu:
            dummy_tgt = dummy_tgt.cuda()

        target = torch.zeros(inp.size()).long()
        target[:, cfg.max_seq_len] = cfg.padding_idx
        target[:, 0:cfg.max_seq_len - 1] = inp[:, 1:cfg.max_seq_len]
        if self.gpu:
            target = target.cuda()

        pred = self.forward(target, inp)


        samples = torch.argmax(pred, dim=-1).view(batch_size, -1)
        print(samples)
        log_prob = F.nll_loss(pred, samples.view(-1), reduction='none').view(batch_size, -1)
        # samples = torch.multinomial(torch.exp(log_prob), 1)
        return samples, log_prob