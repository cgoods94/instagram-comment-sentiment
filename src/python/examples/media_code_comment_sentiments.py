from instagrapi import Client
import os
from dotenv import load_dotenv
import pathlib
import pandas as pd

from instagram import extract_comments_from_code
from preprocessing import remove_urls, remove_emojis, remove_user_tags

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline


def media_code_comment_sentiments(
    media_code: str, model_fp: str, tokenizer_fp: str
) -> pd.DataFrame:
    """Example script that hits the Instagram API
    and returns a DataFrame of the comments for
    any Instagram post when given the media code.

    (e.g. https://instagram.com/p/ --> C28N0_9v7O5 <--)

    Args:
        media_code - the Instagram media code
        model_fp - filepath to the trained model
        tokenizer_fp - filepath to the tokenizer

    Returns:
        a Pandas DataFrame with comments scored using
        a fine-tuned bert-base-uncased model.
    """
    if os.getenv("INSTAGRAM_USER") is None:
        env_path = pathlib.Path("..") / ".." / ".env"
        load_dotenv(dotenv_path=env_path)

    ACCOUNT_USERNAME = os.getenv("INSTAGRAM_USER")
    ACCOUNT_PASSWORD = os.getenv("INSTAGRAM_PASS")

    cl = Client()
    cl.login(ACCOUNT_USERNAME, ACCOUNT_PASSWORD)

    comments_df = extract_comments_from_code(cl, media_code, 100)

    comments_df["text"] = comments_df["text"].apply(lambda x: remove_emojis(x))
    comments_df["text"] = comments_df["text"].apply(lambda x: remove_urls(x))
    comments_df["text"] = comments_df["text"].apply(lambda x: remove_user_tags(x))

    bert_comments_df = comments_df[
        (comments_df["text"] != "") & (comments_df["text"] is not None)
    ].copy()

    model = BertForSequenceClassification.from_pretrained(model_fp)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_fp)

    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Apply the inference
    bert_comments_df["sentiment"] = bert_comments_df["text"].apply(lambda x: nlp(x)[0])

    # Extract the numeric part from the label and map it
    bert_comments_df["sentiment_label"] = bert_comments_df["sentiment"].apply(
        lambda x: x["label"]
    )

    # Extract the score
    bert_comments_df["sentiment_score"] = bert_comments_df["sentiment"].apply(
        lambda x: x["score"]
    )

    return bert_comments_df.sort_values("sentiment_score", ascending=False)
