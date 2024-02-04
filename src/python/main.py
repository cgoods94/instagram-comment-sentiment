import os

if ~(os.getcwd().endswith("src/python/")):
    os.chdir("src/python/")

from examples import media_code_comment_sentiments

model_fp = os.getcwd() + "/model/models/"
tokenizer_fp = os.getcwd() + "/model/tokenizers/"

# Taylor Swift going to the Super Bowl post from ESPN
media_code = "C28N0_9v7O5"

taylor_sb_df = media_code_comment_sentiments(media_code, model_fp, tokenizer_fp)

taylor_sb_df.sentiment_label.value_counts()

taylor_sb_df.loc[
    taylor_sb_df.sentiment_label == "positive", ["text", "sentiment_score"]
].sort_values("sentiment_score", ascending=False)

taylor_sb_df.loc[
    taylor_sb_df.sentiment_label == "negative", ["text", "sentiment_score"]
].sort_values("sentiment_score", ascending=False)

taylor_sb_df.loc[
    taylor_sb_df.sentiment_label == "neutral", ["text", "sentiment_score"]
].sort_values("sentiment_score", ascending=False)
