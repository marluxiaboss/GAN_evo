from transformers import GPT2Tokenizer

from utils.data_loader import GenDataIter


class gpt2_data_loader(GenDataIter):
    def __init__(self, tokenizer):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
