import numpy as np
import pandas as pd

from transformers import BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from torch import tensor
from torch.utils.data import TensorDataset


def gather_and_clean_training_set(
    training_fp: str = "../../data/training_comments_done.xlsx",
    tokenizer_save_path: str = "model/tokenizers/",
) -> TensorDataset:
    """Pulls in the labeled training set from an Excel file
    and converts it into a TensorDataset for BERT to use.

    Args:
        training_fp (str): the filepath to the training data Excel file.
        tokenzier_save_path (str): the save location for the tokenizer

    Returns:
        a torch TensorDataset for use in training the model.

    Side Effects:
        Saves the tokenizer to the specified filepath.
    """
    training_done = pd.read_excel(training_fp)
    training_done = training_done[["text", "label"]].copy()

    training_done["text"] = training_done["text"].fillna("").astype(str)
    training_done["label"] = training_done["label"].map(
        {"positive": 2, "neutral": 1, "negative": 0}
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    encoded_data = _encode_comments(tokenizer, training_done["text"].values)
    input_ids = encoded_data["input_ids"]
    attention_masks = encoded_data["attention_mask"]
    labels = tensor(training_done["label"].values)

    tokenizer.save_pretrained(tokenizer_save_path)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset


def _encode_comments(tokenizer: BertTokenizer, comments: np.ndarray) -> BatchEncoding:
    """A helper function to encode the text of the training comments.

    Args:
        - tokenizer (BertTokenizer): the tokenizer that encodes the
            comments.
        - comments (np.ndarray[str]): an array of comments in string format
            for encoding.

    Returns:
        - a BatchEncoding that yields the input ids and attention mask needed
            for the training Tensor Dataset downstream.

    Raises:
        - ValueError: if the comments are not strings
    """
    if not issubclass(comments.dtype.type, np.object_):
        raise ValueError("Array must contain strings.")

    return tokenizer.batch_encode_plus(
        comments,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
