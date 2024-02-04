from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, random_split

from tqdm import tqdm
import numpy as np

from typing import List


def train_bert_sentiment(
    training_data: TensorDataset,
    tokenizer: BertTokenizer,
    train_size: float = 0.8,
    epochs: int = 4,
    device_type: str = "cpu",
) -> List[float]:
    """Loads a BERT sentiment analysis pipeline.

    Args:
        training_data (TensorDataset): the TensorDataset created by
            gather_and_clean_training_set.
        tokenizer (BertTokenizer): the BERT tokenizer related to
            the training data
        train_size (float): the proportion of training data used
            for training (the rest is used as holdout)
        epochs (int): number of training epochs
        device_type (str): whether you're using ['cpu', 'gpu']

    Returns:
        the loss values from each epoch of training.

    Side Effects:
        Saves the model and the tokenizer to the respective folders
            in the model sub-directory so they can be rendered again
            without re-training.
    """
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,  # Specify the number of sentiment classes
        output_attentions=False,
        output_hidden_states=False,
    )

    device = torch.device(device_type)

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    validation_size = 1 - train_size
    train_dataset, validation_dataset = random_split(
        training_data, [train_size, validation_size]
    )

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=32
    )

    validation_dataloader = DataLoader(
        validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=32
    )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    loss_values = []

    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================
        model.train()  # Put the model into training mode

        total_loss = 0

        # Create a progress bar instance for training
        train_progress_bar = tqdm(
            enumerate(train_dataloader), total=len(train_dataloader), desc="Training"
        )

        for step, batch in train_progress_bar:
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                print("Batch {:>5,}  of  {:>5,}.".format(step, len(train_dataloader)))

            # Unpack the inputs from our dataloader
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

            loss = outputs[0]
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)

        loss_values.append(avg_train_loss)

        print(f"Average training loss: {avg_train_loss}")

        # ========================================
        #               Validation
        # ========================================
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0

        validation_progress_bar = tqdm(
            enumerate(validation_dataloader),
            total=len(validation_dataloader),
            desc="Validation",
        )

        for step, (b_input_ids, b_input_mask, b_labels) in validation_progress_bar:

            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)

            with torch.no_grad():
                # Instructs PyTorch to not compute or store gradients, saving memory and speeding up validation

                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )

                loss = outputs.loss
                logits = outputs.logits

                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to("cpu").numpy()

                total_eval_accuracy += _flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print(f"Validation Accuracy: {avg_val_accuracy}")

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        print(f"Validation Loss: {avg_val_loss}")

        model_save_path = "model/models/"
        tokenizer_save_path = "model/tokenizers/"

        model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        model.config.label2id = {"negative": 0, "neutral": 1, "positive": 2}

        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(tokenizer_save_path)

        return loss_values


def _flat_accuracy(preds, labels):
    """Function to calculate the accuracy of
    predictions vs labels.
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
