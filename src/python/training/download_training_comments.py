import pandas as pd
import time
from instagrapi import Client

from instagram import extract_comments_from_pk
from preprocessing import remove_urls, remove_emojis, remove_user_tags


def download_training_comments(
    client: Client, media_code: str, username: str, amount: int = 100
) -> pd.DataFrame:
    """

    Args:

    Returns:

    """
    my_userid = client.user_id_from_username(username)
    users_i_follow = list(client.user_following(my_userid).keys())

    hi_comment_posts = []

    for user in users_i_follow:

        posts = client.user_medias(user, amount)

        user_posts = []

        for post in posts:

            post_data = post.model_dump()

            # id and comment count
            post_pk = post_data["pk"]
            comm_ct = post_data["comment_count"]

            user_posts.append((post_pk, comm_ct))

        user_posts_df = (
            pd.DataFrame(user_posts, columns=["post_pk", "num_comments"])
            .sort_values("num_comments", ascending=False)
            .head(5)
        )

        hi_comment_posts.append(user_posts_df)

    hi_comment_posts = pd.concat(hi_comment_posts)

    training_comments = []

    for pk in hi_comment_posts["post_pk"]:

        pk_comments = extract_comments_from_pk(client, pk, 20)
        training_comments.append(pk_comments)

        # Rate Limit: 200 / hr = 1 / 18s
        time.sleep(18)

    comments_df = pd.concat(training_comments)

    comments_df["text"] = comments_df["text"].apply(lambda x: remove_emojis(x))
    comments_df["text"] = comments_df["text"].apply(lambda x: remove_urls(x))
    comments_df["text"] = comments_df["text"].apply(lambda x: remove_user_tags(x))

    training_comments = comments_df[
        (comments_df["text"] != "") & (comments_df["text"] is not None)
    ].copy()

    training_comments = training_comments.reset_index()

    return training_comments
