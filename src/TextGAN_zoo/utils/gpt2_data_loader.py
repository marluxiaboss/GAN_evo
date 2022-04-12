import datasets
from transformers import GPT2Tokenizer
import config as cfg

from utils.data_loader import GenDataIter
from utils.text_process import get_tokenlized


class gpt2_data_loader(GenDataIter):
    def __init__(self):
        self.tokens = None
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    @staticmethod
    def tokenize_function(examples):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
        )

    def load_data(self, dataset):
        dataset_path = 'dataset/{}.txt'.format(dataset)
        self.tokens = get_tokenlized(dataset_path)
        dataset = datasets.load_dataset('text', data_files=dataset_path)

        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        return self.prepare(tokenized_dataset)


