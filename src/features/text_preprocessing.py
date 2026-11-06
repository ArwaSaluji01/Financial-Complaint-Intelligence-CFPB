
import html
import re
import unicodedata


def clean_narrative(text):
    """
    Clean a complaint narrative while preserving
    meaningful financial terminology and punctuation.
    """

    if text is None:
        return ""

    text = str(text)

    if not text.strip():
        return ""

    # Normalize Unicode characters.
    text = unicodedata.normalize("NFKC", text)

    # Decode HTML entities.
    text = html.unescape(text)

    # Remove control characters except normal whitespace.
    text = "".join(
        character
        for character in text
        if character.isprintable() or character in "\n\t"
    )

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def make_duplicate_key(text):
    """
    Create a normalized key for exact duplicate detection.

    This is used for grouping, not as model input.
    """

    text = clean_narrative(text)

    return text.casefold().strip()


def is_valid_narrative(text):
    """
    Check whether a complaint narrative is usable.
    """

    cleaned = clean_narrative(text)

    if not cleaned:
        return False

    placeholders = {
        "n/a",
        "na",
        "none",
        "null",
        "not provided",
        "not available",
        "redacted",
        "xxxx"
    }

    if cleaned.casefold() in placeholders:
        return False

    return True
