from instagrapi import Client
import pandas as pd


def extract_comments_from_code(
    client: Client, media_code: str, amount: int = 100
) -> pd.DataFrame:
    """Takes an Instagram media code and dumps
    its comment text, usernames, and timestamps
    into a DataFrame.

    Args:
        client - a primed instagrapi Client
        media_code - the code in an Instagram
        link after the base URI and media type

        e.g.
        Link: 'https://www.instagram.com/p/C2pwhKWvpw9/'
        Media Code: 'C2pwhKWvpw9'

    Returns:
        A Pandas DataFrame with comment data.
    """

    media_id = client.media_pk_from_code(media_code)
    comments = client.media_comments(media_id, amount)

    comm_user_text = []

    for comment in comments:
        comm = comment.model_dump()

        # id, text, username, and timestamp
        comment_id = comm["pk"]
        text = comm["text"]
        user = comm["user"]["username"]
        created_at = comm["created_at_utc"]

        comm_user_text.append((comment_id, text, user, created_at))

    comments_df = pd.DataFrame(
        comm_user_text, columns=["comment_id", "text", "username", "created_at"]
    )

    return comments_df
