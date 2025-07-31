import re
import pandas as pd
from typing import List


def clean_dataframe(
    df: pd.DataFrame,
    column_name: str,
    phrases_to_remove: str | List[str] | None = None,
    remove_empty: bool = True,
    remove_hashtags: bool = True,
    remove_mentions: bool = True,
    remove_urls: bool = True,
):
    """
    Cleans text in a specified DataFrame column by removing links, mentions, and specific phrases.

    Parameters:
    - df: pandas.DataFrame
    - column_name: str, name of the column to clean
    - phrases_to_remove: list of str, phrases to remove from the text
    - remove_empty: flag to remove empty strings from the text
    - remove hashtags: flag to remove hashtags from the text
    - remove_mentions: flag to remove mentions from the text
    - remove_urls: flag to remove urls from the text

    Returns:
    - df: pandas.DataFrame with cleaned text in the specified column
    """
    if phrases_to_remove is None:
        phrases_to_remove = []

    # Compile regex for links and mentions
    url_pattern = re.compile(r"http\S+|www\S+|https\S+|t\.co\S+")
    mention_pattern = re.compile(r"@\w+")
    hashtag_pattern = re.compile(r"#\w+")
    mention_and_hashtag_pattern = re.compile(r"[@#]\w+")

    # Create a single regex for the phrases (escaped and joined by |)
    if isinstance(phrases_to_remove, str):
        phrases_to_remove = [phrases_to_remove]
    phrase_pattern = None
    if phrases_to_remove:
        escaped_phrases = [re.escape(phrase) for phrase in phrases_to_remove]
        phrase_pattern = re.compile(
            r"\b(?:" + "|".join(escaped_phrases) + r")\b", flags=re.IGNORECASE
        )

    def clean_text(text):
        if not isinstance(text, str):
            return text
        if remove_urls:
            text = url_pattern.sub("", text)
        if remove_hashtags and remove_mentions:
            text = mention_and_hashtag_pattern.sub("", text)
        elif remove_hashtags:
            text = hashtag_pattern.sub("", text)
        elif remove_mentions:
            text = mention_pattern.sub("", text)

        if phrase_pattern:
            text = phrase_pattern.sub("", text)
        return re.sub(r"\s+", " ", text).strip()  # Normalize whitespace

    df[column_name] = df[column_name].apply(clean_text)
    if remove_empty:
        return df[df[column_name].str.len() > 0][column_name]
    return df[column_name]


def main():
    # Sample data
    data = {
        "text": [
            "Check this out: https://example.com @john",
            "Totally agree with that. One of the best ideas!",
            "Visit www.site.com for more info. #awesome",
            "@alice That’s what I meant. Just saying. https://t.co/naVew4zgD6",
            "China greenlights homegrown #pneumococcal, #HPV vaccines ",
            "for the market by Pfizer, Merck t.co/YgAZ2q4Xgu",
            "https://t.co/naVew4zgD6",
            "There shouldn't be a blank above me",
        ]
    }

    df = pd.DataFrame(data)

    # Define phrases to remove
    phrases = ["just saying", "one of the best ideas"]

    # Apply the cleaning function
    cleaned_df = clean_dataframe(df, "text", phrases_to_remove=phrases)

    print(cleaned_df)

    # series = pd.Series(["https://test.crit", "hashtag #testme", "mention @person", "delete my neighbour"])
    #
    # df = pd.DataFrame({"text":series})
    # df['idx'] = df.index
    # print(df)
    #
    # df = clean_dataframe(df, "text", remove_list='neighbour')
    # de =
    # print(df)


if __name__ == "__main__":
    main()
