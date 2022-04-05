from models.generator import GPT2Model


class GPT_2(GPT2Model):
    def __init__(self, gen, config):
        super(GPT_2, self).__init__(config)
        self.gen = GPT2Model.from_pretrained('gpt2')

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
        hidden = self.init_hidden(batch_size)

        pred = self.forward(inp, hidden)
        samples = torch.argmax(pred, dim=-1).view(batch_size, -1)
        log_prob = F.nll_loss(pred, samples.view(-1), reduction='none').view(batch_size, -1)
        # samples = torch.multinomial(torch.exp(log_prob), 1)
