import torch
import torch.nn.functional as F

from models.generator import TransformerGenerator
import config as cfg


class SA_DPGAN_G(TransformerGenerator):
    def __init__(self, config):
        super(SA_DPGAN_G, self).__init__(config)

