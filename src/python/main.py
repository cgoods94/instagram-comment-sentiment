from instagrapi import Client
import os
from dotenv import load_dotenv
import pathlib
import pandas as pd

from extract_ig_comments import extract_comments_from_code
from preprocess_comments import contains_url_emoji
from load_embeddings import bert_sentiment

if os.getenv("INSTAGRAM_USER") is None:
    env_path = pathlib.Path("..") / ".." / ".env"
    load_dotenv(dotenv_path=env_path)

ACCOUNT_USERNAME = os.getenv("INSTAGRAM_USER")
ACCOUNT_PASSWORD = os.getenv("INSTAGRAM_PASS")

cl = Client()
cl.login(ACCOUNT_USERNAME, ACCOUNT_PASSWORD)

media_code = "C2pwhKWvpw9"

comments_df = extract_comments_from_code(cl, media_code, 100)

comments_df[["has_url", "has_emoji"]] = (
    comments_df["text"].apply(lambda x: contains_url_emoji(x)).apply(pd.Series)
)

bert_comments_df = comments_df[
    (~(comments_df.has_url)) & (~(comments_df.has_emoji))
].copy()

nlp = bert_sentiment()

bert_comments_df["sentiment"] = bert_comments_df["text"].apply(lambda x: nlp(x)[0])

bert_comments_df["sentiment_label"] = bert_comments_df["sentiment"].apply(
    lambda x: x["label"]
)
bert_comments_df["sentiment_score"] = bert_comments_df["sentiment"].apply(
    lambda x: x["score"]
)
