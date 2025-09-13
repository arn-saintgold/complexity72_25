import re
import pandas as pd
from typing import List


import re
import pandas as pd
from typing import List

def clean_dataframe(
    df: pd.DataFrame,
    column_name: str,
    phrases_to_remove: str | List[str] | None = None,
    remove_empty: bool = True,
    remove_hashtags: bool = False,
    normalize_hashtags: bool = True,
    remove_mentions: bool = False,
    normalize_mentions: bool = True,
    remove_urls: bool = True,
    lowercase: bool = False,
    strip_punctuation: bool = False,
    user_placeholder: str = "user",
) -> pd.DataFrame:
    """
    Cleans text in a specified DataFrame column with flexible preprocessing options.

    Parameters:
    - df: pandas.DataFrame
    - column_name: str, name of the column to clean
    - phrases_to_remove: str or list of str, phrases to remove from the text
    - remove_empty: bool, whether to drop rows where the cleaned text is empty
    - remove_hashtags: bool, remove hashtags entirely (e.g., "#AI" → "")
    - normalize_hashtags: bool, convert hashtags to words (e.g., "#AI" → "AI")
    - remove_mentions: bool, remove mentions entirely (e.g., "@user" → "")
    - normalize_mentions: bool, replace mentions with a placeholder "user"
                          (ignored if remove_mentions=True)
    - remove_urls: bool, remove URLs
    - lowercase: bool, convert text to lowercase
    - strip_punctuation: bool, remove punctuation characters
    - user_placeholder: str, placeholder text for mentions if normalize_mentions=True

    Returns:
        pandas.DataFrame: Copy of df with an additional column "Clean<column_name>"
                          containing the cleaned text.
    """

    if phrases_to_remove is None:
        phrases_to_remove = []

    # Compile regex patterns
    url_pattern = re.compile(r"http\S+|www\S+|https\S+|t\.co\S+")
    mention_pattern = re.compile(r"@\w+")
    hashtag_pattern = re.compile(r"#(\w+)")
    punct_pattern = re.compile(r"[^\w\s]")

    # Create a single regex for custom phrases
    phrase_pattern = None
    if phrases_to_remove:
        escaped_phrases = []
        for phrase in (phrases_to_remove if isinstance(phrases_to_remove, list) else [phrases_to_remove]):
            if re.match(r"^\w+$", phrase):  # only word chars → safe for \b
                escaped_phrases.append(r"\b" + re.escape(phrase) + r"\b")
            else:
                escaped_phrases.append(re.escape(phrase))
        phrase_pattern = re.compile("|".join(escaped_phrases), flags=re.IGNORECASE)

    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return text

        # Remove URLs
        if remove_urls:
            text = url_pattern.sub("", text)

        # Mentions
        if remove_mentions:
            text = mention_pattern.sub("", text)
        elif normalize_mentions:
            text = mention_pattern.sub(user_placeholder, text)

        # Hashtags
        if remove_hashtags:
            text = hashtag_pattern.sub("", text)  # remove entirely
        elif normalize_hashtags:
            text = hashtag_pattern.sub(r"\1", text)  # keep word, drop '#'

        # Custom phrase removal
        if phrase_pattern:
            text = phrase_pattern.sub("", text)

        # Lowercase
        if lowercase:
            text = text.lower()

        # Strip punctuation
        if strip_punctuation:
            text = punct_pattern.sub("", text)

        # Normalize whitespace
        return re.sub(r"\s+", " ", text).strip()
    
    new_col_name = "Clean" + column_name
    df[new_col_name] = df[column_name].apply(clean_text)
    
    if remove_empty:
        return df[df[new_col_name].str.len() > 0].copy()
    
    return df.copy()


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
