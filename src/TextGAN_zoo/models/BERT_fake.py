import numpy as np
import pandas as pd
import config as cfg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
import wandb

wandb.init(project="bert-fake-detection", entity="hdasilva")

from models.generator import LSTMGenerator
from utils.bp_encoder import get_encoder
import torch.nn.functional as F

"""
inspiration from:
https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
"""


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


class BERT_fake:
    def __init__(self):
        # Define pretrained tokenizer and model
        self.model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.bpe = get_encoder()
        # Initialize the dataset for training
        data = pd.read_csv('dataset/image_coco_fake_true.csv')

        # ----- 1. Preprocess data -----#
        # Preprocess data
        X = list(data["text"])
        y = list(data["label"])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        X_train_tokenized = self.tokenizer(X_train, padding=True, truncation=True, max_length=512)
        X_val_tokenized = self.tokenizer(X_val, padding=True, truncation=True, max_length=512)

        self.train_dataset = Dataset(X_train_tokenized, y_train)
        self.val_dataset = Dataset(X_val_tokenized, y_val)

        # Define Trainer parameters
        """
        args = TrainingArguments(
            output_dir="output",
            evaluation_strategy="steps",
            #evaluation_strategy="epoch",
            eval_steps=1,
            max_steps=20,
            per_device_train_batch_size=int(cfg.batch_size / 2),
            per_device_eval_batch_size=int(cfg.batch_size / 2),
            num_train_epochs=cfg.d_epoch,
            seed=0,
            save_total_limit=1,
            load_best_model_at_end=True,
        )
        """
        args = TrainingArguments(
            output_dir="output",
            evaluation_strategy="steps",
            # evaluation_strategy="epoch",
            eval_steps=40,
            per_device_train_batch_size=int(cfg.batch_size / 2),
            per_device_eval_batch_size=int(cfg.batch_size / 2),
            gradient_accumulation_steps=16,
            num_train_epochs=cfg.d_epoch,
            seed=0,
            save_steps=40,
            save_total_limit=1,
            load_best_model_at_end=True,
            log_level="critical",
            report_to="wandb"
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

    def load_model(self):
        # Load trained model
        model_path = "output/checkpoint-1000"
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

    def compute_metrics(self, p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)
        print("label: {}".format(labels))
        print("pred: {}".format(pred))
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)
        print("ACCCCC")
        print(accuracy)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def evaluate(self, path=None, epoch=0):
        if not path:
            self.trainer.evaluate()
        else:
            # Initialize the dataset for training
            image_coco_fake_true_path = cfg.save_samples_root + 'image_coco_fake_true' + str(epoch) + '.csv'
            data = pd.read_csv(image_coco_fake_true_path)
            # ----- 1. Preprocess data -----#
            # Preprocess data
            X = list(data["text"])
            y = list(data["label"])
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1.0)
            X_val_tokenized = self.tokenizer(X_val, padding=True, truncation=True, max_length=512)
            val_dataset = Dataset(X_val_tokenized, y_val)
            self.trainer.evaluate(val_dataset)

    def fake_detection_train(self):
        # Train pre-trained model
        self.trainer.train()
        """
        loss = self.trainer.evaluate()
        print("LOSS")
        print(loss)
        """

        """
        print("SAMPLES")
        print(samples)
        samples = samples.tolist()
        samples = [self.bpe.decode(sample) for sample in samples]
        X_test_tokenized = self.tokenizer(samples, padding=True, truncation=True, max_length=512)
        test_dataset = Dataset(X_test_tokenized)

        # Make prediction
        raw_pred, _, _ = self.trainer.predict(test_dataset)

        # Preprocess raw predictions
        y_pred = np.argmax(raw_pred, axis=1)
        print("SAMPLES")
        print(samples)
        print("y_pred")
        print(y_pred)
        print("y_raw")
        print(raw_pred)
        """

    def getReward(self, samples, training_bin):
        samples = samples.tolist()
        samples = [self.bpe.decode(sample) for sample in samples]
        samples_cut = []
        for sample in samples:
            sample_tokenized = self.ntlk_tokenizer.tokenize(sample.split())
            # sample_tokenized = sample
            sample_tokenized = sample_tokenized[:cfg.max_seq_len]
            samples_cut.append(sample_tokenized)
        X_test_tokenized = self.tokenizer(samples_cut, padding=True, truncation=True, max_length=512)
        test_dataset = Dataset(X_test_tokenized)

        # Make prediction
        raw_pred, _, _ = self.trainer.predict(test_dataset)

        # Preprocess raw predictions
        y_pred = np.argmax(raw_pred, axis=1)
        raw_pred = torch.tensor(raw_pred)
        y_soft = F.softmax(raw_pred, dim=1)
        sentence_rewards = y_soft[:, 1]
        """
        print("SAMPLES")
        print(samples)
        print("y_pred")
        print(y_pred)
        print("y_raw")
        print(raw_pred)
        """

        # fill the bins for the histogram
        for y in y_pred:
            training_bin[y] += 1
        sentence_rewards = torch.tensor(sentence_rewards, requires_grad=False)
        return sentence_rewards
