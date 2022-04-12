import config as cfg
from models.generator import TransformerGenerator
import torch.nn.functional as F


class GPT_2(TransformerGenerator):
    def __init__(self):
        config = cfg.GPT2Config(vocab_size_or_config_json_file=50257,
                                n_positions=1024,
                                n_ctx=1024,
                                n_embd=768,
                                n_layer=12,
                                n_head=12,
                                layer_norm_epsilon=1e-5,
                                initializer_range=0.02)
        super(GPT_2, self).__init__(config)
    """
    def sample_sequence(self, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0,
                        device='cuda', sample=True, sample_pos2=False):
        # TODO: should I assume that the input is already tokenized or maybe I can tokenize here
    """

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


        logits, past = self(inp)
        pred = F.softmax(logits, dim=-1)
        samples = self.sample_sequence(cfg.max_seq_len - 1, start_token=cfg.start_letter,
                                                     batch_size=batch_size, temperature=0.7,
                                                     top_k=1, sample=False)
        log_prob = F.nll_loss(pred.view(-1, cfg.GPT2Config().vocab_size), samples.view(-1),
                              reduction='none').view(batch_size, -1)

        return samples, log_prob
