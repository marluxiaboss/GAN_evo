import torch
import torch.nn.functional as F

from models.generator import TransformerGenerator
import config as cfg


class SA_DPGAN_G(TransformerGenerator):
    def __init__(self, config):
        super(SA_DPGAN_G, self).__init__(config)

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

        pred = self.forward(inp)
        samples = torch.argmax(pred, dim=-1).view(batch_size, -1)
        log_prob = F.nll_loss(pred, samples.view(-1), reduction='none').view(batch_size, -1)
        # samples = torch.multinomial(torch.exp(log_prob), 1)

        return samples, log_prob

