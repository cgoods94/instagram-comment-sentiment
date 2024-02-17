import os

if ("src/python" not in os.getcwd()):
    os.chdir("src/python/")

from examples import media_code_comment_sentiments

model_fp = os.getcwd() + "/model/models/"
tokenizer_fp = os.getcwd() + "/model/tokenizers/"

# Example Post: @jason.kelce had an elite strategy for Gridiron Gauntlet 😭
media_code = "C28N0_9v7O5"

jason_kelce_df = media_code_comment_sentiments(media_code, model_fp, tokenizer_fp)

jason_kelce_df.sentiment_label.value_counts()

jason_kelce_df.loc[
    jason_kelce_df.sentiment_label == "positive", ["text", "sentiment_score"]
].sort_values("sentiment_score", ascending=False)

jason_kelce_df.loc[
    jason_kelce_df.sentiment_label == "negative", ["text", "sentiment_score"]
].sort_values("sentiment_score", ascending=False)

jason_kelce_df.loc[
    jason_kelce_df.sentiment_label == "neutral", ["text", "sentiment_score"]
].sort_values("sentiment_score", ascending=False)
