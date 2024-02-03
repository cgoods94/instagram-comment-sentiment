from instagrapi import Client
import os
from dotenv import load_dotenv
import pathlib
import pandas as pd

from extract_ig_comments import extract_comments_from_code
from preprocess_comments import remove_urls, remove_emojis
from load_embeddings import bert_sentiment


def taylor_swift_superbowl() -> pd.DataFrame:
    """Example script that hits the Instagram API
    and returns a DataFrame of the comments from
    ESPN's post about Taylor Swift flying from Japan
    to attend the Super Bowl next week.

    Args: None

    Returns:
        a Pandas DataFrame with comments scored using
        bert-base-uncased model.
    """
    if os.getenv("INSTAGRAM_USER") is None:
        env_path = pathlib.Path("..") / ".." / ".env"
        load_dotenv(dotenv_path=env_path)

    ACCOUNT_USERNAME = os.getenv("INSTAGRAM_USER")
    ACCOUNT_PASSWORD = os.getenv("INSTAGRAM_PASS")

    cl = Client()
    cl.login(ACCOUNT_USERNAME, ACCOUNT_PASSWORD)

    # Instagram post about Taylor Swift flying from Japan to the Super Bowl
    media_code = "C2sd8ksuAbL"

    comments_df = extract_comments_from_code(cl, media_code, 100)

    comments_df['text'] = comments_df['text'].apply(lambda x: remove_emojis(x))
    comments_df['text'] = comments_df['text'].apply(lambda x: remove_urls(x))

    bert_comments_df = comments_df[(comments_df['text'] != '') &
                              (comments_df['text'] is not None)].copy()

    nlp = bert_sentiment()

    bert_comments_df["sentiment"] = bert_comments_df["text"].apply(lambda x: nlp(x)[0])

    bert_comments_df["sentiment_label"] = bert_comments_df["sentiment"].apply(
        lambda x: x["label"]
    )
    bert_comments_df["sentiment_score"] = bert_comments_df["sentiment"].apply(
        lambda x: x["score"]
    )

    return bert_comments_df
