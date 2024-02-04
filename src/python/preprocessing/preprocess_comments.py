import re
from typing import Tuple


def contains_url_emoji(text: str) -> Tuple[bool, bool]:
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


def remove_urls(text: str) -> str:
    """Instead of flagging URLs, this removes them from the string instead.

    Args:
        text - the text of the Instagram comment

    Returns:
        the comment text with URLs removed.
    """
    # Regex pattern for detecting URLs
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

    return re.sub(url_pattern, "", text)


def remove_emojis(text: str) -> str:
    """Instead of flagging emojis, this removes them from the string instead.

    Args:
        text - the text of the Instagram comment

    Returns:
        the comment text with emojis removed.
    """
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
    return re.sub(emoji_pattern, "", text)


def remove_user_tags(text: str) -> str:
    """Removes user tags (@'s) from the comment text.

    Args:
        text - the text of the Instagram comment.

    Returns:
        the text string with user tags removed.
    """
    return re.sub(r"@\w+", "", text)
