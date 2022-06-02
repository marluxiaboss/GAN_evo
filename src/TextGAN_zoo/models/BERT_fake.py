import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback

from models.generator import LSTMGenerator

"""
inspiration from:
https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
"""
class BERT_fake(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(BERT_fake, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)

        # Define pretrained tokenizer and model
        self.model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(sefl.model_name, num_labels=2)

        # Initialize the dataset for training
        data = pd.read_csv("train.csv")

        # ----- 1. Preprocess data -----#
        # Preprocess data
        X = list(data["text"])
        y = list(data["label"])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        X_train_tokenized = self.tokenizer(X_train, padding=True, truncation=True, max_length=512)
        X_val_tokenized = self.tokenizer(X_val, padding=True, truncation=True, max_length=512)

        # Create torch dataset
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels=None):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                if self.labels:
                    item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.encodings["input_ids"])

        self.train_dataset = Dataset(X_train_tokenized, y_train)
        self.val_dataset = Dataset(X_val_tokenized, y_val)

        # Define Trainer parameters
        args = TrainingArguments(
            output_dir="output",
            evaluation_strategy="steps",
            eval_steps=500,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=1,
            seed=0,
            load_best_model_at_end=True,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

    def compute_metrics(self, p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def fake_detection_train(self):

        # Train pre-trained model
        self.trainer.train()
        loss = self.trainer.evaluate()
        print("LOSS")
        print(loss)
        y






