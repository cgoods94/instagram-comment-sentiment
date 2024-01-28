import re
from typing import Tuple


def contains_url_emoji(text) -> Tuple[bool, bool]:
    """Flags a piece of text as containing a URL and/or
    an emoji.

    Args:
        text - the text of the Instagram comment.

    Returns:
        a Tuple of two bools - has_url and has_emoji
    """
    # Regex pattern for detecting URLs
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    # Regex pattern for detecting emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    return bool(url_pattern.search(text)), bool(emoji_pattern.search(text))
